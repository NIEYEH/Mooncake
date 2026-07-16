#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

PROFILE=${PROFILE:-full}
BIN=${BIN:-${REPO_ROOT}/build/mooncake-store/benchmarks/gds_store_bench}
MASTER_SERVER=${MASTER_SERVER:-127.0.0.1:50051}
METADATA_SERVER=${METADATA_SERVER:-http://127.0.0.1:8080/metadata}
LOCAL_HOSTNAME=${LOCAL_HOSTNAME:-$(hostname):50071}
GDS_SEGMENT=${GDS_SEGMENT:-gds_pool_0}
GDS_DEVICE=${GDS_DEVICE:-}
GPU_ID=${GPU_ID:-0}
TENANT_ID=${TENANT_ID:-default}

# The current mounted path is expected to top out below this value. It is a
# reporting reference and a capacity guard, not a performance pass threshold.
EXPECTED_CEILING_GIB_S=${EXPECTED_CEILING_GIB_S:-10}
MAX_TRANSIENT_WRITE_GIB=${MAX_TRANSIENT_WRITE_GIB:-128}

DURATION_SEC=${DURATION_SEC:-15}
WRITE_DURATION_SEC=${WRITE_DURATION_SEC:-10}
WARMUP_SEC=${WARMUP_SEC:-3}
READ_DATASET_GIB=${READ_DATASET_GIB:-4}
MAX_DATASET_OBJECTS=${MAX_DATASET_OBJECTS:-16384}

STRESS_VALUE_SIZE=${STRESS_VALUE_SIZE:-4194304}
STRESS_DATASET_GIB=${STRESS_DATASET_GIB:-16}
STRESS_THREADS=${STRESS_THREADS:-16}
STRESS_BATCH_SIZE=${STRESS_BATCH_SIZE:-4}
STRESS_READ_DURATION_SEC=${STRESS_READ_DURATION_SEC:-300}
STRESS_WRITE_DURATION_SEC=${STRESS_WRITE_DURATION_SEC:-30}
STRESS_MIXED_DURATION_SEC=${STRESS_MIXED_DURATION_SEC:-120}
STRESS_MIXED_READ_RATIO=${STRESS_MIXED_READ_RATIO:-70}

RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/gds-bench-results/${RUN_ID}}
CSV_PATH=${OUTPUT_DIR}/results.csv
LOG_PATH=${OUTPUT_DIR}/run.log

usage() {
  cat <<'EOF'
Usage: run_gds_store_bench.sh [--full|--quick|--stress-only]

The default --full profile runs correctness, size/concurrency/batch/mixed
matrices, and single-node stress tests. All GDS writes are destructive to the
registered test pool. Do not use a system disk or a production namespace.

Important environment variables:
  BIN                         gds_store_bench executable
  MASTER_SERVER               Master host:port
  METADATA_SERVER             TENT metadata URL/connection string
  LOCAL_HOSTNAME              Non-loopback host:port; host must match accessor
  GDS_SEGMENT                 Preferred registered GDS segment
  GDS_DEVICE                  Optional local block path for a preflight check
  GPU_ID                      CUDA device ID
  GDS_BENCH_CONFIRM=YES       Non-interactive destructive confirmation
  EXPECTED_CEILING_GIB_S=10   Reference bandwidth, not a pass threshold
  MAX_TRANSIENT_WRITE_GIB=128 Capacity guard for timed write/mixed stress
  OUTPUT_DIR                  Result directory
EOF
}

case "${1:-}" in
  ""|--full) PROFILE=full ;;
  --quick) PROFILE=quick ;;
  --stress-only) PROFILE=stress ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac

die() {
  echo "ERROR: $*" >&2
  exit 1
}

is_positive_number() {
  awk -v value="$1" 'BEGIN { exit !(value + 0 > 0) }'
}

min_int() {
  if (( $1 < $2 )); then
    echo "$1"
  else
    echo "$2"
  fi
}

safe_duration() {
  local write_fraction=$1
  awk -v cap="${MAX_TRANSIENT_WRITE_GIB}" \
      -v bw="${EXPECTED_CEILING_GIB_S}" \
      -v fraction="${write_fraction}" \
      'BEGIN {
         seconds = int(cap / (bw * fraction));
         if (seconds < 1) seconds = 1;
         print seconds;
       }'
}

dataset_objects() {
  local value_size=$1
  local threads=$2
  local batch_size=$3
  local count=$((READ_DATASET_GIB * 1024 * 1024 * 1024 / value_size))
  local minimum=$((threads * batch_size * 8))
  if (( count > MAX_DATASET_OBJECTS )); then count=${MAX_DATASET_OBJECTS}; fi
  if (( count < minimum )); then count=${minimum}; fi
  echo "${count}"
}

stress_dataset_objects() {
  local count=$((STRESS_DATASET_GIB * 1024 * 1024 * 1024 / STRESS_VALUE_SIZE))
  local minimum=$((STRESS_THREADS * STRESS_BATCH_SIZE * 8))
  if (( count < minimum )); then count=${minimum}; fi
  echo "${count}"
}

if [[ ${GDS_BENCH_CONFIRM:-} != YES ]]; then
  if [[ -t 0 ]]; then
    echo "WARNING: this writes to the registered raw GDS SSD test pool."
    read -r -p "Type YES to continue: " answer
    [[ ${answer} == YES ]] || die "confirmation declined"
  else
    die "set GDS_BENCH_CONFIRM=YES for non-interactive execution"
  fi
fi

[[ -x ${BIN} ]] || die "benchmark executable not found: ${BIN}"
[[ ${LOCAL_HOSTNAME} == *:* ]] || die "LOCAL_HOSTNAME must include a port"
[[ ${LOCAL_HOSTNAME%%:*} != 127.0.0.1 ]] || die "LOCAL_HOSTNAME must not be loopback"
[[ ${LOCAL_HOSTNAME%%:*} != localhost ]] || die "LOCAL_HOSTNAME must not be loopback"
is_positive_number "${EXPECTED_CEILING_GIB_S}" || die "invalid EXPECTED_CEILING_GIB_S"
is_positive_number "${MAX_TRANSIENT_WRITE_GIB}" || die "invalid MAX_TRANSIENT_WRITE_GIB"

if [[ -n ${GDS_DEVICE} && ! -b ${GDS_DEVICE} ]]; then
  die "GDS_DEVICE is not a block device: ${GDS_DEVICE}"
fi
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
nvidia-smi -i "${GPU_ID}" >/dev/null || die "GPU ${GPU_ID} is unavailable"

export MC_USE_TENT=${MC_USE_TENT:-1}
export MC_TENT_CONF=${MC_TENT_CONF:-'{"transports":{"gds":{"enable":true}}}'}

mkdir -p "${OUTPUT_DIR}"
touch "${LOG_PATH}"

COMMON_ARGS=(
  "--local_hostname=${LOCAL_HOSTNAME}"
  "--metadata_server=${METADATA_SERVER}"
  "--master_server=${MASTER_SERVER}"
  "--tenant_id=${TENANT_ID}"
  "--gpu_id=${GPU_ID}"
  "--memory_replica_num=0"
  "--nof_replica_num=0"
  "--gds_replica_num=1"
  "--preferred_gds_segments=${GDS_SEGMENT}"
)

CURRENT_CASE=preflight
CURRENT_PREFIX=

on_exit() {
  local rc=$?
  if (( rc != 0 )); then
    echo "FAILED case=${CURRENT_CASE} prefix=${CURRENT_PREFIX:-none}" | tee -a "${LOG_PATH}"
    echo "A killed timed write can leave objects under that prefix; inspect Master before rerunning." | tee -a "${LOG_PATH}"
  fi
}
trap on_exit EXIT

run_bench() {
  local case_name=$1
  local write_csv=$2
  shift 2
  local args=("${COMMON_ARGS[@]}" "--case_name=${case_name}" "$@")
  if [[ ${write_csv} == yes ]]; then
    args+=("--csv_path=${CSV_PATH}")
  fi
  CURRENT_CASE=${case_name}
  CURRENT_PREFIX=
  for arg in "${args[@]}"; do
    if [[ ${arg} == --key_prefix=* ]]; then CURRENT_PREFIX=${arg#*=}; fi
  done

  {
    echo
    echo "===== ${case_name} ====="
    printf 'COMMAND:'
    printf ' %q' "${BIN}" "${args[@]}"
    printf '\n'
  } | tee -a "${LOG_PATH}"

  set +e
  "${BIN}" "${args[@]}" 2>&1 | tee -a "${LOG_PATH}"
  local rc=${PIPESTATUS[0]}
  set -e
  if (( rc != 0 )); then
    echo "Case ${case_name} failed with rc=${rc}" | tee -a "${LOG_PATH}"
    return "${rc}"
  fi
}

prepare_dataset() {
  local label=$1 size=$2 count=$3 threads=$4 batch=$5 prefix=$6
  run_bench "${label}_prepare" no \
    --operation=prepare \
    "--key_prefix=${prefix}" \
    "--value_size=${size}" \
    "--num_objects=${count}" \
    "--threads=${threads}" \
    "--batch_size=${batch}" \
    --warmup_sec=0 \
    --allow_destructive_gds=YES
}

verify_case() {
  local label=$1 size=$2 count=$3 threads=$4 batch=$5
  local prefix="${RUN_ID}-${label}"
  run_bench "${label}" yes \
    --operation=verify \
    "--key_prefix=${prefix}" \
    "--value_size=${size}" \
    "--num_objects=${count}" \
    "--threads=${threads}" \
    "--batch_size=${batch}" \
    --warmup_sec=0 \
    --allow_destructive_gds=YES \
    --cleanup_after_run=true
}

write_case() {
  local label=$1 size=$2 threads=$3 batch=$4 duration=$5
  local prefix="${RUN_ID}-${label}"
  run_bench "${label}" yes \
    --operation=write \
    "--key_prefix=${prefix}" \
    "--value_size=${size}" \
    "--threads=${threads}" \
    "--batch_size=${batch}" \
    "--warmup_sec=${WARMUP_SEC}" \
    "--duration_sec=${duration}" \
    --allow_destructive_gds=YES \
    --cleanup_after_run=true
}

read_case() {
  local label=$1 size=$2 threads=$3 batch=$4 duration=$5
  local count prefix prep_threads prep_batch
  count=$(dataset_objects "${size}" "${threads}" "${batch}")
  prefix="${RUN_ID}-${label}"
  prep_threads=$(min_int "${threads}" 4)
  prep_batch=$(min_int "${batch}" 8)
  prepare_dataset "${label}" "${size}" "${count}" \
    "${prep_threads}" "${prep_batch}" "${prefix}"
  run_bench "${label}" yes \
    --operation=read \
    "--key_prefix=${prefix}" \
    "--value_size=${size}" \
    "--num_objects=${count}" \
    "--threads=${threads}" \
    "--batch_size=${batch}" \
    "--warmup_sec=${WARMUP_SEC}" \
    "--duration_sec=${duration}" \
    --cleanup_after_run=true
}

mixed_case() {
  local label=$1 size=$2 threads=$3 batch=$4 read_ratio=$5 duration=$6
  local count prefix
  count=$(dataset_objects "${size}" "${threads}" "${batch}")
  prefix="${RUN_ID}-${label}"
  run_bench "${label}" yes \
    --operation=mixed \
    "--key_prefix=${prefix}" \
    "--value_size=${size}" \
    "--num_objects=${count}" \
    "--threads=${threads}" \
    "--batch_size=${batch}" \
    "--read_ratio=${read_ratio}" \
    "--warmup_sec=${WARMUP_SEC}" \
    "--duration_sec=${duration}" \
    --prepare_before_run=true \
    --allow_destructive_gds=YES \
    --cleanup_after_run=true
}

run_correctness() {
  echo "Running correctness suite" | tee -a "${LOG_PATH}"
  verify_case correctness_4k_single 4096 64 1 1
  verify_case correctness_1m_batch 1048576 32 4 4
  verify_case correctness_4m_batch 4194304 16 2 2
  verify_case correctness_16m_single 16777216 8 1 1

  local prefix="${RUN_ID}-check-existing"
  prepare_dataset check_existing 4194304 32 4 4 "${prefix}"
  run_bench check_existing yes \
    --operation=check \
    "--key_prefix=${prefix}" \
    --value_size=4194304 \
    --num_objects=32 \
    --threads=4 \
    --batch_size=4 \
    --cleanup_after_run=true
}

run_performance_matrices() {
  local sizes threads batches size thread batch
  if [[ ${PROFILE} == quick ]]; then
    sizes=(4096 1048576 4194304)
    threads=(1 4)
    batches=(1 4)
  else
    sizes=(4096 65536 1048576 4194304 16777216)
    threads=(1 2 4 8)
    batches=(1 4 8 16)
  fi

  for size in "${sizes[@]}"; do
    write_case "size_write_${size}" "${size}" 1 1 "${WRITE_DURATION_SEC}"
    read_case "size_read_${size}" "${size}" 1 1 "${DURATION_SEC}"
  done

  for thread in "${threads[@]}"; do
    write_case "concurrency_write_t${thread}" 4194304 "${thread}" 1 \
      "${WRITE_DURATION_SEC}"
    read_case "concurrency_read_t${thread}" 4194304 "${thread}" 1 \
      "${DURATION_SEC}"
  done

  for batch in "${batches[@]}"; do
    write_case "batch_write_b${batch}" 1048576 4 "${batch}" \
      "${WRITE_DURATION_SEC}"
    read_case "batch_read_b${batch}" 1048576 4 "${batch}" \
      "${DURATION_SEC}"
  done

  mixed_case mixed_r90 4194304 4 1 90 "${DURATION_SEC}"
  mixed_case mixed_r70 4194304 4 1 70 "${DURATION_SEC}"
  mixed_case mixed_r50 4194304 4 1 50 "${DURATION_SEC}"
}

run_stress() {
  local count prefix prep_threads prep_batch
  local safe_write_sec actual_write_sec write_fraction safe_mixed_sec
  local actual_mixed_sec

  count=$(stress_dataset_objects)
  prefix="${RUN_ID}-stress-read"
  prep_threads=$(min_int "${STRESS_THREADS}" 4)
  prep_batch=$(min_int "${STRESS_BATCH_SIZE}" 8)
  prepare_dataset stress_read "${STRESS_VALUE_SIZE}" "${count}" \
    "${prep_threads}" "${prep_batch}" "${prefix}"
  run_bench stress_read yes \
    --operation=read \
    "--key_prefix=${prefix}" \
    "--value_size=${STRESS_VALUE_SIZE}" \
    "--num_objects=${count}" \
    "--threads=${STRESS_THREADS}" \
    "--batch_size=${STRESS_BATCH_SIZE}" \
    "--warmup_sec=${WARMUP_SEC}" \
    "--duration_sec=${STRESS_READ_DURATION_SEC}" \
    --latency_sample_every=10 \
    --cleanup_after_run=true

  safe_write_sec=$(safe_duration 1.0)
  actual_write_sec=$(min_int "${STRESS_WRITE_DURATION_SEC}" "${safe_write_sec}")
  echo "Stress write duration=${actual_write_sec}s (capacity guard=${safe_write_sec}s)" | tee -a "${LOG_PATH}"
  write_case stress_write "${STRESS_VALUE_SIZE}" "${STRESS_THREADS}" \
    "${STRESS_BATCH_SIZE}" "${actual_write_sec}"

  write_fraction=$(awk -v ratio="${STRESS_MIXED_READ_RATIO}" \
    'BEGIN { print (100 - ratio) / 100.0 }')
  safe_mixed_sec=$(safe_duration "${write_fraction}")
  actual_mixed_sec=$(min_int "${STRESS_MIXED_DURATION_SEC}" "${safe_mixed_sec}")
  echo "Stress mixed duration=${actual_mixed_sec}s (capacity guard=${safe_mixed_sec}s)" | tee -a "${LOG_PATH}"
  mixed_case stress_mixed "${STRESS_VALUE_SIZE}" "${STRESS_THREADS}" \
    "${STRESS_BATCH_SIZE}" "${STRESS_MIXED_READ_RATIO}" \
    "${actual_mixed_sec}"
}

print_summary() {
  [[ -s ${CSV_PATH} ]] || return
  echo | tee -a "${LOG_PATH}"
  echo "===== RESULT SUMMARY =====" | tee -a "${LOG_PATH}"
  awk -F, -v ceiling="${EXPECTED_CEILING_GIB_S}" '
    NR == 1 { next }
    {
      utilization = ($13 / ceiling) * 100.0;
      printf "%-34s op=%-7s size=%8.3fMiB t=%2d b=%2d bw=%7.3fGiB/s util=%6.1f%% p99=%9.1fus cpu=%6.1f%% fail=%s\n",
             $1, $4, $5 / 1048576.0, $6, $7, $13, utilization, $21, $18, $11;
    }
  ' "${CSV_PATH}" | tee -a "${LOG_PATH}"
  echo "Reference ceiling: ${EXPECTED_CEILING_GIB_S} GiB/s (informational only)" | tee -a "${LOG_PATH}"
}

{
  echo "Mooncake single-node GDS Store benchmark"
  echo "run_id=${RUN_ID} profile=${PROFILE}"
  echo "binary=${BIN}"
  echo "master=${MASTER_SERVER} metadata=${METADATA_SERVER}"
  echo "local_hostname=${LOCAL_HOSTNAME} segment=${GDS_SEGMENT} gpu=${GPU_ID}"
  echo "reference_ceiling=${EXPECTED_CEILING_GIB_S}GiB/s transient_write_limit=${MAX_TRANSIENT_WRITE_GIB}GiB"
  echo "output_dir=${OUTPUT_DIR}"
} | tee -a "${LOG_PATH}"

run_correctness
if [[ ${PROFILE} != stress ]]; then
  run_performance_matrices
fi
if [[ ${PROFILE} != quick ]]; then
  run_stress
fi
print_summary

CURRENT_CASE=complete
CURRENT_PREFIX=
echo "PASS: all requested cases completed" | tee -a "${LOG_PATH}"
echo "CSV: ${CSV_PATH}"
echo "Log: ${LOG_PATH}"
