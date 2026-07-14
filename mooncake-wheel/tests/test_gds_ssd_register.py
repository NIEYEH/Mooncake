import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


def load_register_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "mooncake"
        / "mooncake_gds_ssd_register.py"
    )
    fake_store = types.ModuleType("mooncake.store")
    fake_store.MooncakeGdsSsdRegister = object
    previous_store = sys.modules.get("mooncake.store")
    sys.modules["mooncake.store"] = fake_store
    try:
        spec = importlib.util.spec_from_file_location(
            "mooncake_gds_ssd_register_test_module", module_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_store is None:
            sys.modules.pop("mooncake.store", None)
        else:
            sys.modules["mooncake.store"] = previous_store


gds_register = load_register_module()


class FakeRegistrar:
    def __init__(self, result=0, error=None):
        self.result = result
        self.error = error

    def real_register(self, *args):
        if self.error is not None:
            raise self.error
        return self.result


class GdsSsdRegistrationPlanValidationTest(unittest.TestCase):
    def make_args(self, **overrides):
        values = {
            "master_server_address": "127.0.0.1:50051",
            "traddr": None,
            "trsvcid": "4420",
            "trtype": "tcp",
            "nqn": None,
            "nsid": 1,
            "device": "/dev/nvme0n1",
            "stable_path": None,
            "segment_uri": None,
            "segment_name": "gds_test",
            "client_host": "host-a",
            "namespace_id": "namespace-test",
            "base": 0,
            "size": 8192,
            "device_size": 8192,
            "block_size": 4096,
            "allocation_alignment": 4096,
            "metadata_reserved_bytes": 0,
            "gpu_device_ids": "0,1",
            "numa_node": -1,
            "skip_nvme_connect": True,
            "dry_run": True,
            "force": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_build_plan_does_not_mutate_stable_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stable_path = Path(temp_dir) / "gds_test"
            plan = gds_register.build_registration_plan(
                self.make_args(stable_path=str(stable_path))
            )

            self.assertTrue(plan["manage_stable_path"])
            self.assertFalse(os.path.lexists(stable_path))

    def test_rejects_conflicting_segment_uri_and_stable_path(self):
        with self.assertRaisesRegex(
            gds_register.GdsSsdRegisterError, "stable_path must match"
        ):
            gds_register.build_registration_plan(
                self.make_args(
                    segment_uri="block:///dev/mooncake/path-a",
                    stable_path="/dev/mooncake/path-b",
                )
            )

    def test_rejects_empty_block_uri_and_invalid_alignment(self):
        with self.assertRaisesRegex(
            gds_register.GdsSsdRegisterError, "must not be empty"
        ):
            gds_register.build_registration_plan(
                self.make_args(segment_uri="block://")
            )
        with self.assertRaisesRegex(
            gds_register.GdsSsdRegisterError, "positive multiple"
        ):
            gds_register.build_registration_plan(
                self.make_args(allocation_alignment=2048)
            )


@unittest.skipUnless(os.name == "posix", "stable device paths are Linux-only")
class GdsSsdRegisterRollbackTest(unittest.TestCase):
    def make_plan(self, device, stable_path):
        return {
            "master_server_address": "127.0.0.1:50051",
            "segment_name": "gds_test",
            "client_host": "host-a",
            "segment_uri": f"block://{stable_path}",
            "namespace_id": "namespace-test",
            "base": 0,
            "size": 4096,
            "device_size": 4096,
            "block_size": 4096,
            "allocation_alignment": 4096,
            "metadata_reserved_bytes": 0,
            "gpu_device_ids": [],
            "numa_node": -1,
            "device": str(device),
            "stable_path": str(stable_path),
            "manage_stable_path": True,
            "nqn": None,
            "nsid": None,
        }

    def test_success_keeps_new_stable_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            device = root / "device"
            device.write_bytes(b"device")
            stable_path = root / "dev" / "mooncake" / "gds_test"
            plan = self.make_plan(device, stable_path)
            gds_register.MooncakeGdsSsdRegister = lambda: FakeRegistrar()

            result = gds_register.register(plan, dry_run=False)

            self.assertEqual(result, gds_register.OPERATION_OK)
            self.assertTrue(stable_path.is_symlink())
            self.assertEqual(stable_path.resolve(), device.resolve())

    def test_failed_registration_removes_new_stable_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            device = root / "device"
            device.write_bytes(b"device")
            stable_path = root / "stable"
            plan = self.make_plan(device, stable_path)
            gds_register.MooncakeGdsSsdRegister = lambda: FakeRegistrar(result=1)

            result = gds_register.register(plan, dry_run=False)

            self.assertEqual(result, 1)
            self.assertFalse(os.path.lexists(stable_path))

    def test_failed_forced_registration_restores_previous_symlink(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_device = root / "old-device"
            new_device = root / "new-device"
            old_device.write_bytes(b"old")
            new_device.write_bytes(b"new")
            stable_path = root / "stable"
            stable_path.symlink_to(old_device)
            plan = self.make_plan(new_device, stable_path)
            gds_register.MooncakeGdsSsdRegister = lambda: FakeRegistrar(result=1)

            result = gds_register.register(plan, dry_run=False, force=True)

            self.assertEqual(result, 1)
            self.assertTrue(stable_path.is_symlink())
            self.assertEqual(stable_path.resolve(), old_device.resolve())

    def test_exception_rolls_back_stable_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            device = root / "device"
            device.write_bytes(b"device")
            stable_path = root / "stable"
            plan = self.make_plan(device, stable_path)
            gds_register.MooncakeGdsSsdRegister = lambda: FakeRegistrar(
                error=RuntimeError("rpc failed")
            )

            with self.assertRaisesRegex(RuntimeError, "rpc failed"):
                gds_register.register(plan, dry_run=False)

            self.assertFalse(os.path.lexists(stable_path))

    def test_dry_run_does_not_create_stable_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            device = root / "device"
            device.write_bytes(b"device")
            stable_path = root / "stable"
            plan = self.make_plan(device, stable_path)

            result = gds_register.register(plan, dry_run=True)

            self.assertEqual(result, gds_register.OPERATION_OK)
            self.assertFalse(os.path.lexists(stable_path))


if __name__ == "__main__":
    unittest.main()
