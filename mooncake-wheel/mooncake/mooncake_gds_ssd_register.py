#!/usr/bin/env python3
"""Register per-host GDS SSD accessors with a Mooncake master."""

import argparse
import hashlib
import json
import logging
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import fcntl
    import struct
except ImportError:  # pragma: no cover - Linux-only path
    fcntl = None
    struct = None

from mooncake.store import MooncakeGdsSsdRegister


OPERATION_OK = 0
BLKGETSIZE64 = 0x80081272
BLKSSZGET = 0x1268


class GdsSsdRegisterError(RuntimeError):
    pass


def parse_size(value: str) -> int:
    text = str(value).strip()
    if not text:
        raise argparse.ArgumentTypeError("size value must not be empty")
    units = {
        "k": 1024,
        "kb": 1024,
        "m": 1024 ** 2,
        "mb": 1024 ** 2,
        "g": 1024 ** 3,
        "gb": 1024 ** 3,
        "t": 1024 ** 4,
        "tb": 1024 ** 4,
    }
    lower = text.lower()
    for suffix, multiplier in sorted(units.items(), key=lambda item: -len(item[0])):
        if lower.endswith(suffix):
            number = lower[: -len(suffix)]
            break
    else:
        number = lower
        multiplier = 1
    if not number.isdigit():
        raise argparse.ArgumentTypeError(f"invalid size value: {value}")
    return int(number) * multiplier


def parse_gpu_ids(value: str) -> List[int]:
    if not value:
        return []
    ids: List[int] = []
    seen = set()
    for item in value.split(","):
        if item == "":
            raise GdsSsdRegisterError("gpu_device_ids contains an empty item")
        if not item.isdigit():
            raise GdsSsdRegisterError("gpu_device_ids must contain integers")
        gpu_id = int(item)
        if gpu_id in seen:
            raise GdsSsdRegisterError(f"duplicate gpu id: {gpu_id}")
        seen.add(gpu_id)
        ids.append(gpu_id)
    return ids


def run_command(command: Sequence[str], check: bool = True) -> subprocess.CompletedProcess:
    logging.debug("Running command: %s", " ".join(command))
    try:
        result = subprocess.run(
            list(command),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise GdsSsdRegisterError(
            f"command not found: {command[0]}; install nvme-cli or pass --device/--skip_nvme_connect"
        ) from exc
    if check and result.returncode != 0:
        output = (result.stderr or result.stdout).strip()
        raise GdsSsdRegisterError(
            f"command failed ({' '.join(command)}): {output or result.returncode}"
        )
    return result


def discover_nqn(args: argparse.Namespace) -> Optional[str]:
    if args.nqn:
        return args.nqn
    if args.skip_nvme_connect or not args.traddr:
        return None

    result = run_command(
        [
            "nvme",
            "discover",
            "-t",
            args.trtype,
            "-a",
            args.traddr,
            "-s",
            str(args.trsvcid),
            "-o",
            "json",
        ]
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise GdsSsdRegisterError("nvme discover did not return valid JSON") from exc

    records = payload.get("Records") or payload.get("records") or []
    candidates = sorted(
        {
            record.get("subnqn") or record.get("subsysnqn")
            for record in records
            if record.get("subnqn") or record.get("subsysnqn")
        }
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise GdsSsdRegisterError("nvme discover found no NQN; pass --nqn explicitly")
    raise GdsSsdRegisterError(
        "nvme discover found multiple NQNs; pass --nqn explicitly: "
        + ", ".join(candidates)
    )


def connect_namespace(args: argparse.Namespace, nqn: Optional[str]) -> None:
    if args.dry_run or args.skip_nvme_connect:
        return
    if not args.traddr:
        return
    if not nqn:
        raise GdsSsdRegisterError("--nqn is required when running nvme connect")

    command = [
        "nvme",
        "connect",
        "-t",
        args.trtype,
        "-a",
        args.traddr,
        "-s",
        str(args.trsvcid),
        "-n",
        nqn,
    ]
    result = run_command(command, check=False)
    if result.returncode == 0:
        return
    output = (result.stderr or result.stdout).lower()
    if "already" in output and "connect" in output:
        logging.info("NVMe namespace is already connected")
        return
    raise GdsSsdRegisterError(
        f"nvme connect failed: {(result.stderr or result.stdout).strip()}"
    )


def list_nvme_devices() -> List[Dict[str, Any]]:
    result = run_command(["nvme", "list", "-o", "json"])
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise GdsSsdRegisterError("nvme list did not return valid JSON") from exc
    devices = payload.get("Devices") or payload.get("devices") or []
    if not isinstance(devices, list):
        return []
    return devices


def field_int(record: Dict[str, Any], *names: str) -> Optional[int]:
    for name in names:
        value = record.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def find_device(args: argparse.Namespace, nqn: Optional[str]) -> str:
    if args.device:
        return args.device
    if args.dry_run:
        raise GdsSsdRegisterError("dry-run without --device cannot infer a block device")
    devices = list_nvme_devices()
    candidates: List[str] = []
    for record in devices:
        record_nqn = record.get("SubsystemNQN") or record.get("subsystemnqn")
        if nqn and record_nqn and record_nqn != nqn:
            continue
        if args.nsid is not None:
            record_nsid = field_int(record, "NameSpace", "NSID", "nsid")
            if record_nsid is not None and record_nsid != args.nsid:
                continue
        path = record.get("DevicePath") or record.get("device") or record.get("Name")
        if path:
            candidates.append(path)

    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise GdsSsdRegisterError("could not infer block device; pass --device")
    raise GdsSsdRegisterError(
        "multiple block devices match; pass --device explicitly: "
        + ", ".join(candidates)
    )


def is_block_device(path: str) -> bool:
    try:
        return stat.S_ISBLK(os.stat(path).st_mode)
    except FileNotFoundError:
        return False


def ioctl_u64(path: str, request: int) -> int:
    if fcntl is None or struct is None:
        raise GdsSsdRegisterError("block ioctl is only supported on Linux")
    with open(path, "rb", buffering=0) as device:
        data = bytearray(8)
        fcntl.ioctl(device.fileno(), request, data, True)
    return struct.unpack("Q", data)[0]


def ioctl_u32(path: str, request: int) -> int:
    if fcntl is None or struct is None:
        raise GdsSsdRegisterError("block ioctl is only supported on Linux")
    with open(path, "rb", buffering=0) as device:
        data = bytearray(4)
        fcntl.ioctl(device.fileno(), request, data, True)
    return struct.unpack("I", data)[0]


def sysfs_block_name(path: str) -> str:
    return Path(os.path.realpath(path)).name


def read_first_sysfs_value(path: str, names: Iterable[str]) -> Optional[str]:
    block_name = sysfs_block_name(path)
    root = Path("/sys/class/block") / block_name
    for name in names:
        candidate = root / name
        try:
            value = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return None


def build_fallback_namespace_id(args: argparse.Namespace, nqn: Optional[str]) -> Optional[str]:
    if not nqn and args.nsid is None:
        return None
    parts = [f"trtype={args.trtype}"]
    if args.traddr:
        parts.append(f"traddr={args.traddr}")
    parts.append(f"trsvcid={args.trsvcid}")
    if nqn:
        parts.append(f"nqn={nqn}")
    if args.nsid is not None:
        parts.append(f"nsid={args.nsid}")
    return ";".join(parts)


def resolve_namespace_id(path: str, args: argparse.Namespace, nqn: Optional[str]) -> str:
    if args.namespace_id:
        return args.namespace_id
    value = read_first_sysfs_value(
        path,
        ["wwid", "device/wwid", "device/nguid", "device/uuid", "device/eui"],
    )
    if value:
        return value
    fallback = build_fallback_namespace_id(args, nqn)
    if fallback:
        logging.warning("Falling back to transport-derived namespace_id")
        return fallback
    raise GdsSsdRegisterError("could not read namespace_id; pass --namespace_id")


def default_segment_name(namespace_id: str) -> str:
    digest = hashlib.sha256(namespace_id.encode("utf-8")).hexdigest()[:12]
    return f"gds_{digest}"


def prepare_stable_path(device: str, stable_path: str, force: bool, dry_run: bool) -> None:
    if os.path.realpath(device) == os.path.realpath(stable_path):
        return
    if dry_run:
        logging.info("dry-run: would point %s at %s", stable_path, device)
        return

    parent = os.path.dirname(stable_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.lexists(stable_path):
        if os.path.islink(stable_path):
            current = os.path.realpath(stable_path)
            expected = os.path.realpath(device)
            if current == expected:
                return
            if not force:
                raise GdsSsdRegisterError(
                    f"stable path {stable_path} points to {current}, not {expected}; use --force"
                )
            os.unlink(stable_path)
        else:
            if os.path.realpath(stable_path) == os.path.realpath(device):
                return
            raise GdsSsdRegisterError(
                f"stable path {stable_path} exists and is not a symlink"
            )
    os.symlink(device, stable_path)


def build_registration_plan(args: argparse.Namespace) -> Dict[str, Any]:
    nqn = discover_nqn(args)
    connect_namespace(args, nqn)
    device = find_device(args, nqn)

    if not args.dry_run and not is_block_device(device):
        raise GdsSsdRegisterError(f"{device} is not a block device")

    if args.device_size is not None:
        device_size = args.device_size
    elif args.dry_run and args.size is not None:
        device_size = args.size + args.base
    else:
        device_size = ioctl_u64(device, BLKGETSIZE64)

    if args.block_size is not None:
        block_size = args.block_size
    elif args.dry_run:
        raise GdsSsdRegisterError("dry-run requires --block_size")
    else:
        block_size = ioctl_u32(device, BLKSSZGET)

    if block_size <= 0:
        raise GdsSsdRegisterError("block_size must be positive")
    if args.base < 0:
        raise GdsSsdRegisterError("base must be non-negative")
    if device_size <= args.base:
        raise GdsSsdRegisterError("device_size must be larger than base")

    segment_size = args.size if args.size is not None else device_size - args.base
    if segment_size <= 0:
        raise GdsSsdRegisterError("segment size must be positive")
    if args.base + segment_size > device_size:
        raise GdsSsdRegisterError("segment range exceeds device size")

    alignment = args.allocation_alignment or max(block_size, 4096)
    if args.base % alignment != 0:
        raise GdsSsdRegisterError("base must be aligned to allocation_alignment")

    namespace_id = resolve_namespace_id(device, args, nqn)
    segment_name = args.segment_name or default_segment_name(namespace_id)
    if args.segment_uri:
        segment_uri = args.segment_uri
        if not segment_uri.startswith("block://"):
            raise GdsSsdRegisterError("segment_uri must use block://")
        stable_path = args.stable_path or segment_uri[len("block://") :]
        if args.stable_path:
            prepare_stable_path(device, stable_path, args.force, args.dry_run)
    else:
        stable_path = args.stable_path or f"/dev/mooncake/{segment_name}"
        segment_uri = f"block://{stable_path}"
        prepare_stable_path(device, stable_path, args.force, args.dry_run)

    return {
        "master_server_address": args.master_server_address,
        "segment_name": segment_name,
        "client_host": args.client_host or socket.gethostname(),
        "segment_uri": segment_uri,
        "namespace_id": namespace_id,
        "base": args.base,
        "size": segment_size,
        "device_size": device_size,
        "block_size": block_size,
        "allocation_alignment": alignment,
        "metadata_reserved_bytes": args.metadata_reserved_bytes,
        "gpu_device_ids": parse_gpu_ids(args.gpu_device_ids),
        "numa_node": args.numa_node,
        "device": device,
        "stable_path": stable_path,
        "nqn": nqn,
        "nsid": args.nsid,
    }


def register(plan: Dict[str, Any], dry_run: bool) -> int:
    print(json.dumps(plan, indent=2, sort_keys=True))
    if dry_run:
        return OPERATION_OK
    registrar = MooncakeGdsSsdRegister()
    return registrar.real_register(
        plan["master_server_address"],
        plan["segment_name"],
        plan["client_host"],
        plan["segment_uri"],
        plan["namespace_id"],
        plan["base"],
        plan["size"],
        plan["device_size"],
        plan["block_size"],
        plan["allocation_alignment"],
        plan["metadata_reserved_bytes"],
        plan["gpu_device_ids"],
        plan["numa_node"],
    )


def print_status(master_server_address: str) -> int:
    registrar = MooncakeGdsSsdRegister()
    segments = registrar.list_segments(master_server_address)
    print(json.dumps(list(segments), indent=2, sort_keys=True))
    return OPERATION_OK


def unregister(args: argparse.Namespace) -> int:
    if not args.segment_name:
        raise GdsSsdRegisterError("--segment_name is required for --unregister")
    client_host = args.client_host or socket.gethostname()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "master_server_address": args.master_server_address,
                    "segment_name": args.segment_name,
                    "client_host": client_host,
                    "alive": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return OPERATION_OK
    registrar = MooncakeGdsSsdRegister()
    return registrar.real_unregister(
        args.master_server_address, args.segment_name, client_host
    )


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a Mooncake GDS SSD accessor")
    parser.add_argument("--master_server_address", required=True)
    parser.add_argument("--traddr", help="NVMe-oF target address")
    parser.add_argument("--trsvcid", default="4420")
    parser.add_argument("--trtype", default="tcp", choices=["tcp", "rdma", "TCP", "RDMA"])
    parser.add_argument("--nqn")
    parser.add_argument("--nsid", type=int)
    parser.add_argument("--device", help="Local block device path, e.g. /dev/nvme0n1")
    parser.add_argument("--stable_path", help="Stable local path, default /dev/mooncake/<segment_name>")
    parser.add_argument("--segment_uri", help="Explicit block:// URI")
    parser.add_argument("--segment_name")
    parser.add_argument("--client_host")
    parser.add_argument("--namespace_id")
    parser.add_argument("--base", type=parse_size, default=0)
    parser.add_argument("--size", type=parse_size)
    parser.add_argument("--device_size", type=parse_size)
    parser.add_argument("--block_size", type=int)
    parser.add_argument("--allocation_alignment", type=int)
    parser.add_argument("--metadata_reserved_bytes", type=parse_size, default=0)
    parser.add_argument("--gpu_device_ids", default="")
    parser.add_argument("--numa_node", type=int, default=-1)
    parser.add_argument("--skip_nvme_connect", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--unregister", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args(argv)
    args.trtype = args.trtype.lower()
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        if args.status:
            return print_status(args.master_server_address)
        if args.unregister:
            return unregister(args)
        plan = build_registration_plan(args)
        return register(plan, args.dry_run)
    except GdsSsdRegisterError as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())