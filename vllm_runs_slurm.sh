#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

CONFIG_FILE=""
ASSUME_YES=false
INSIDE_ALLOCATION=false
GENERATE_CONFIG=false
RESUME_MODE=false
RESUME_DIR=""

MODEL_CONFIGS=()
INPUT_LENS=(1024)
OUTPUT_LENS=(512)
CONCURRENCIES=(16)
NUM_PROMPTS=100
REPEATS=1
DATASET_NAME="random"
BACKEND="openai"

HOST="0.0.0.0"
PORT=8000
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=""
EXTRA_SERVER_ARGS=""
EXTRA_BENCH_ARGS=""

CONTAINER_IMAGE="docker://rocm/vllm-dev:nightly_main_20260318"
CONTAINER_INSTANCE_NAME=""
SINGULARITY_CACHEDIR_DEFAULT="/var/tmp/${USER}/singularity-cache"
SINGULARITY_TMPDIR_DEFAULT="/var/tmp/${USER}/singularity-tmp"

SLURM_PARTITION="256C8G1H_MI355X_Ubuntu22"
SLURM_GPUS=1
SLURM_TIME="00:45:00"
SLURM_MEM="0"
SLURM_RESERVATION=""
SLURM_ACCOUNT=""
SLURM_QOS=""
SLURM_JOB_NAME="vllm-bench"
SLURM_EXTRA_ARGS=""

BASE_OUT_DIR="${SCRIPT_DIR}/benchmark_results_slurm"
RUN_DIR=""
SERVER_PID=""
SERVER_RC=0
INSTANCE_STARTED_BY_SCRIPT=false
TOTAL_BENCHMARKS=0
COMPLETED_BENCHMARKS=0
FAILED_BENCHMARKS=0
RESUMED_BENCHMARKS=0
RESULTS_TRACKER=""

declare -A COMPLETED_RUNS=()

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

fail() {
  printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2
  exit 1
}

print_progress() {
  local remaining=0
  if (( TOTAL_BENCHMARKS >= COMPLETED_BENCHMARKS + FAILED_BENCHMARKS )); then
    remaining=$((TOTAL_BENCHMARKS - COMPLETED_BENCHMARKS - FAILED_BENCHMARKS))
  fi
  printf '[%s] Progress: completed=%d failed=%d remaining=%d total=%d\n' \
    "$(date +%H:%M:%S)" \
    "${COMPLETED_BENCHMARKS}" \
    "${FAILED_BENCHMARKS}" \
    "${remaining}" \
    "${TOTAL_BENCHMARKS}"
}

parse_model_path() {
  echo "$1" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1}'
}

parse_precisions() {
  echo "$1" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}'
}

parse_tp_sizes() {
  echo "$1" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3}'
}

parse_model_extra() {
  echo "$1" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $4); print $4}'
}

normalize_list_array() {
  local -n arr_ref="$1"
  local joined normalized
  joined="${arr_ref[*]}"
  normalized="$(tr ',' ' ' <<< "${joined}")"
  read -r -a arr_ref <<< "${normalized}"
}

model_short_name() {
  basename "$1" | sed 's/[^a-zA-Z0-9._-]/_/g'
}

benchmark_tag() {
  local model_path="$1"
  local precision="$2"
  local tp="$3"
  local concurrency="$4"
  local input_len="$5"
  local output_len="$6"
  local rep="$7"
  printf '%s/%s_tp%s/c%s_in%s_out%s_rep%s' \
    "$(model_short_name "${model_path}")" \
    "${precision}" \
    "${tp}" \
    "${concurrency}" \
    "${input_len}" \
    "${output_len}" \
    "${rep}"
}

map_precision_to_vllm_args() {
  case "$1" in
    bf16|bfloat16) echo "--dtype bfloat16" ;;
    fp16|float16) echo "--dtype float16" ;;
    fp8|fp8_w8a8) echo "--dtype auto --quantization fp8" ;;
    awq) echo "--dtype auto --quantization awq" ;;
    gptq) echo "--dtype auto --quantization gptq" ;;
    auto) echo "--dtype auto" ;;
    *) echo "--dtype $1" ;;
  esac
}

shell_join() {
  local out=()
  local arg
  for arg in "$@"; do
    out+=("$(printf '%q' "$arg")")
  done
  printf '%s' "${out[*]}"
}

show_help() {
  cat <<EOF
Usage:
  ${SCRIPT_NAME} -c CONFIG_FILE [--yes]
  ${SCRIPT_NAME} --generate-config [OUTPUT_FILE]

The script can be launched from the AAC login node. It will request a Slurm
allocation when needed, start a Singularity instance on the compute node,
launch vLLM inside the container, run one or more benchmarks, and clean up.

Options:
  -c, --config FILE        Load configuration from FILE
  -r, --resume [DIR]       Resume a previous run directory, or latest if DIR omitted
  -g, --generate-config    Write a sample config file
  -y, --yes                Skip the confirmation prompt
  -h, --help               Show this help

Config file notes:
  MODEL_CONFIGS entries use:
    "model_path | precisions | tp_sizes | extra_vllm_args"

Example:
  ${SCRIPT_NAME} -c ./vllm_runs_slurm.conf
EOF
}

generate_sample_config() {
  local output_file="${1:-vllm_runs_slurm.conf}"
  cat > "${output_file}" <<'EOF'
# Example config for vllm_runs_slurm.sh

MODEL_CONFIGS=(
  "meta-llama/Llama-3.1-8B-Instruct | bf16 | 1"
)

INPUT_LENS=(1024)
OUTPUT_LENS=(512)
CONCURRENCIES=(16)
NUM_PROMPTS=100
REPEATS=1
DATASET_NAME="random"
BACKEND="openai"

HOST="0.0.0.0"
PORT=8000
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=""
EXTRA_SERVER_ARGS=""
EXTRA_BENCH_ARGS=""

CONTAINER_IMAGE="docker://rocm/vllm-dev:nightly_main_20260318"
CONTAINER_INSTANCE_NAME=""

SLURM_PARTITION="256C8G1H_MI355X_Ubuntu22"
SLURM_GPUS=1
SLURM_TIME="00:45:00"
SLURM_MEM="0"
SLURM_RESERVATION=""
SLURM_ACCOUNT=""
SLURM_QOS=""
SLURM_JOB_NAME="vllm-bench"
SLURM_EXTRA_ARGS=""

BASE_OUT_DIR="./benchmark_results_slurm"
EOF
  log "Wrote sample config to ${output_file}"
}

find_latest_run_dir() {
  local base_dir="$1"
  find "${base_dir}" -maxdepth 1 -mindepth 1 -type d -name 'run_*' 2>/dev/null | sort | tail -n 1
}

stop_server() {
  if [[ -n "${SERVER_PID}" ]]; then
    log "Stopping vLLM server (pid ${SERVER_PID})"
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
    SERVER_PID=""
  fi
}

stop_instance() {
  if [[ -n "${CONTAINER_INSTANCE_NAME}" && "${INSTANCE_STARTED_BY_SCRIPT}" == "true" ]]; then
    log "Stopping Singularity instance ${CONTAINER_INSTANCE_NAME}"
    singularity instance stop "${CONTAINER_INSTANCE_NAME}" >/dev/null 2>&1 || true
    INSTANCE_STARTED_BY_SCRIPT=false
  fi
}

cleanup() {
  set +e
  stop_server
  stop_instance
}
trap cleanup EXIT INT TERM

validate_config() {
  [[ ${#MODEL_CONFIGS[@]} -gt 0 ]] || fail "MODEL_CONFIGS is empty."
  command -v singularity >/dev/null 2>&1 || fail "singularity is required on the compute node."
  command -v curl >/dev/null 2>&1 || fail "curl is required."
  normalize_list_array INPUT_LENS
  normalize_list_array OUTPUT_LENS
  normalize_list_array CONCURRENCIES
  [[ ${#INPUT_LENS[@]} -gt 0 ]] || fail "INPUT_LENS is empty."
  [[ ${#OUTPUT_LENS[@]} -gt 0 ]] || fail "OUTPUT_LENS is empty."
  [[ ${#CONCURRENCIES[@]} -gt 0 ]] || fail "CONCURRENCIES is empty."
  for entry in "${MODEL_CONFIGS[@]}"; do
    local model_path precision_list tp_list
    model_path="$(parse_model_path "${entry}")"
    precision_list="$(parse_precisions "${entry}")"
    tp_list="$(parse_tp_sizes "${entry}")"
    [[ -n "${model_path}" ]] || fail "Invalid MODEL_CONFIGS entry: ${entry}"
    [[ -n "${precision_list}" ]] || fail "Missing precision list: ${entry}"
    [[ -n "${tp_list}" ]] || fail "Missing TP list: ${entry}"
  done
}

compute_total_benchmarks() {
  local entry
  TOTAL_BENCHMARKS=0
  for entry in "${MODEL_CONFIGS[@]}"; do
    local prec_count tp_count
    read -ra precisions <<< "$(parse_precisions "${entry}" | tr ',' ' ')"
    read -ra tp_sizes <<< "$(parse_tp_sizes "${entry}" | tr ',' ' ')"
    prec_count="${#precisions[@]}"
    tp_count="${#tp_sizes[@]}"
    TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + prec_count * tp_count * ${#CONCURRENCIES[@]} * ${#INPUT_LENS[@]} * ${#OUTPUT_LENS[@]} * REPEATS))
  done
}

load_completed_runs() {
  local tracker_file="$1"
  [[ -f "${tracker_file}" ]] || return 0

  local line run_name rc
  while IFS= read -r line; do
    [[ "${line}" == RUN* ]] && continue
    [[ "${line}" =~ ^-+$ ]] && continue
    [[ -z "${line}" ]] && continue
    run_name="$(awk '{print $1}' <<< "${line}")"
    rc="$(awk '{print $2}' <<< "${line}")"
    if [[ -n "${run_name}" && "${rc}" == "0" ]]; then
      COMPLETED_RUNS["${run_name}"]=1
      RESUMED_BENCHMARKS=$((RESUMED_BENCHMARKS + 1))
    fi
  done < "${tracker_file}"
}

is_run_completed() {
  local run_key="$1"
  [[ -n "${COMPLETED_RUNS[${run_key}]:-}" ]]
}

all_benchmarks_completed_for_combo() {
  local model_path="$1"
  local precision="$2"
  local tp="$3"
  local concurrency input_len output_len rep run_key

  for concurrency in "${CONCURRENCIES[@]}"; do
    for input_len in "${INPUT_LENS[@]}"; do
      for output_len in "${OUTPUT_LENS[@]}"; do
        for rep in $(seq 1 "${REPEATS}"); do
          run_key="$(benchmark_tag "${model_path}" "${precision}" "${tp}" "${concurrency}" "${input_len}" "${output_len}" "${rep}")"
          if ! is_run_completed "${run_key}"; then
            return 1
          fi
        done
      done
    done
  done

  return 0
}

wait_for_server() {
  local server_log="$1"
  local timeout_secs="${2:-900}"
  local waited=0
  while (( waited < timeout_secs )); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      log "vLLM server is healthy on port ${PORT}"
      return 0
    fi
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      wait "${SERVER_PID}" || SERVER_RC=$?
      log "Server log tail:"
      tail -n 50 "${server_log}" || true
      fail "vLLM server exited before becoming healthy."
    fi
    sleep 5
    waited=$((waited + 5))
  done

  log "Server log tail:"
  tail -n 50 "${server_log}" || true
  fail "Timed out waiting for vLLM server health endpoint."
}

prepare_container_env() {
  export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${SINGULARITY_CACHEDIR_DEFAULT}}"
  export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-${SINGULARITY_TMPDIR_DEFAULT}}"
  mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"
  log "SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR}"
  log "SINGULARITY_TMPDIR=${SINGULARITY_TMPDIR}"

  if [[ -z "${CONTAINER_INSTANCE_NAME}" ]]; then
    CONTAINER_INSTANCE_NAME="vllm-bench-${USER}-${SLURM_JOB_ID:-$$}"
  fi

  if singularity instance list | awk '{print $1}' | grep -Fxq "${CONTAINER_INSTANCE_NAME}"; then
    log "Reusing existing instance ${CONTAINER_INSTANCE_NAME}"
    INSTANCE_STARTED_BY_SCRIPT=false
    return
  fi

  log "Starting Singularity instance ${CONTAINER_INSTANCE_NAME} from ${CONTAINER_IMAGE}"
  singularity instance start "${CONTAINER_IMAGE}" "${CONTAINER_INSTANCE_NAME}"
  INSTANCE_STARTED_BY_SCRIPT=true
}

start_server_for_combo() {
  local model_path="$1"
  local precision="$2"
  local tp_size="$3"
  local model_extra="$4"
  local combo_dir="$5"
  local server_log="${combo_dir}/server.log"
  local server_cmd_file="${combo_dir}/server_cmd.txt"
  local prec_args
  local -a serve_cmd
  local extra_arr=()
  local model_extra_arr=()

  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    fail "Port ${PORT} is already serving traffic on this node."
  fi

  prec_args="$(map_precision_to_vllm_args "${precision}")"
  read -ra serve_cmd <<< "${prec_args}"
  if [[ -n "${EXTRA_SERVER_ARGS}" ]]; then
    read -ra extra_arr <<< "${EXTRA_SERVER_ARGS}"
  fi
  if [[ -n "${model_extra}" ]]; then
    read -ra model_extra_arr <<< "${model_extra}"
  fi

  serve_cmd=(
    singularity exec "instance://${CONTAINER_INSTANCE_NAME}"
    vllm serve "${model_path}"
    "${serve_cmd[@]}"
    --tensor-parallel-size "${tp_size}"
    --host "${HOST}"
    --port "${PORT}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    "${extra_arr[@]}"
    "${model_extra_arr[@]}"
  )
  if [[ -n "${MAX_MODEL_LEN}" ]]; then
    serve_cmd+=(--max-model-len "${MAX_MODEL_LEN}")
  fi

  printf '%s\n' "$(shell_join "${serve_cmd[@]}")" > "${server_cmd_file}"
  log "Starting server for model=${model_path} precision=${precision} tp=${tp_size}"
  "${serve_cmd[@]}" > "${server_log}" 2>&1 &
  SERVER_PID=$!
  wait_for_server "${server_log}"
}

run_benchmark_once() {
  local model_path="$1"
  local precision="$2"
  local tp="$3"
  local concurrency="$4"
  local input_len="$5"
  local output_len="$6"
  local rep="$7"
  local combo_dir="$8"
  local tag="c${concurrency}_in${input_len}_out${output_len}_rep${rep}"
  local bench_log="${combo_dir}/bench_${tag}.log"
  local bench_cmd_file="${combo_dir}/bench_${tag}_cmd.txt"
  local run_key
  local -a bench_cmd
  local extra_arr=()

  run_key="$(benchmark_tag "${model_path}" "${precision}" "${tp}" "${concurrency}" "${input_len}" "${output_len}" "${rep}")"

  if is_run_completed "${run_key}"; then
    log "Skipping completed benchmark ${run_key}"
    print_progress
    return 0
  fi

  if [[ -n "${EXTRA_BENCH_ARGS}" ]]; then
    read -ra extra_arr <<< "${EXTRA_BENCH_ARGS}"
  fi

  bench_cmd=(
    singularity exec "instance://${CONTAINER_INSTANCE_NAME}"
    vllm bench serve
    --backend "${BACKEND}"
    --base-url "http://127.0.0.1:${PORT}"
    --model "${model_path}"
    --dataset-name "${DATASET_NAME}"
    --num-prompts "${NUM_PROMPTS}"
    --max-concurrency "${concurrency}"
    --random-input-len "${input_len}"
    --random-output-len "${output_len}"
    "${extra_arr[@]}"
  )

  printf '%s\n' "$(shell_join "${bench_cmd[@]}")" > "${bench_cmd_file}"
  log "Running benchmark ${tag}"
  {
    echo "STARTED: $(date -Is)"
    echo "RUN_KEY: ${run_key}"
    echo "COMMAND: $(shell_join "${bench_cmd[@]}")"
    echo "PROGRESS_BEFORE: completed=${COMPLETED_BENCHMARKS} failed=${FAILED_BENCHMARKS} remaining=$((TOTAL_BENCHMARKS - COMPLETED_BENCHMARKS - FAILED_BENCHMARKS - 1)) total=${TOTAL_BENCHMARKS}"
    echo ""
  } > "${bench_log}"

  if "${bench_cmd[@]}" 2>&1 | tee -a "${bench_log}"; then
    COMPLETED_BENCHMARKS=$((COMPLETED_BENCHMARKS + 1))
    COMPLETED_RUNS["${run_key}"]=1
    printf '%-90s %-4s %s\n' "${run_key}" "0" "$(date -Is)" >> "${RESULTS_TRACKER}"
  else
    FAILED_BENCHMARKS=$((FAILED_BENCHMARKS + 1))
    printf '%-90s %-4s %s\n' "${run_key}" "1" "$(date -Is)" >> "${RESULTS_TRACKER}"
    print_progress
    return 1
  fi

  {
    echo ""
    echo "FINISHED: $(date -Is)"
    echo "PROGRESS_AFTER: completed=${COMPLETED_BENCHMARKS} failed=${FAILED_BENCHMARKS} remaining=$((TOTAL_BENCHMARKS - COMPLETED_BENCHMARKS - FAILED_BENCHMARKS)) total=${TOTAL_BENCHMARKS}"
  } >> "${bench_log}"

  print_progress
}

run_inside_allocation() {
  local run_name="run_${TIMESTAMP}"
  if [[ "${RESUME_MODE}" == "true" ]]; then
    if [[ "${RESUME_DIR}" == "__auto__" || -z "${RESUME_DIR}" ]]; then
      RESUME_DIR="$(find_latest_run_dir "${BASE_OUT_DIR}")"
    fi
    [[ -n "${RESUME_DIR}" && -d "${RESUME_DIR}" ]] || fail "Resume directory not found."
    RUN_DIR="${RESUME_DIR}"
  else
    RUN_DIR="${BASE_OUT_DIR}/${run_name}"
    mkdir -p "${RUN_DIR}"
  fi
  RESULTS_TRACKER="${RUN_DIR}/results_tracker.txt"

  log "Running on compute node $(hostname) in Slurm job ${SLURM_JOB_ID:-unknown}"
  log "Output directory: ${RUN_DIR}"

  validate_config
  compute_total_benchmarks
  if [[ "${RESUME_MODE}" == "true" ]]; then
    load_completed_runs "${RESULTS_TRACKER}"
    COMPLETED_BENCHMARKS="${RESUMED_BENCHMARKS}"
    log "Loaded completed benchmark points from resume state: ${RESUMED_BENCHMARKS}"
  else
    {
      printf '%-90s %-4s %s\n' "RUN" "RC" "TIMESTAMP"
      printf '%.0s-' {1..125}
      printf '\n'
    } > "${RESULTS_TRACKER}"
  fi
  prepare_container_env
  log "Planned benchmark points: ${TOTAL_BENCHMARKS}"
  print_progress

  local entry model_path model_extra model_name precision tp
  for entry in "${MODEL_CONFIGS[@]}"; do
    model_path="$(parse_model_path "${entry}")"
    model_extra="$(parse_model_extra "${entry}")"
    model_name="$(model_short_name "${model_path}")"
    mkdir -p "${RUN_DIR}/${model_name}"

    read -ra precisions <<< "$(parse_precisions "${entry}" | tr ',' ' ')"
    read -ra tp_sizes <<< "$(parse_tp_sizes "${entry}" | tr ',' ' ')"

    for precision in "${precisions[@]}"; do
      for tp in "${tp_sizes[@]}"; do
        local combo_dir="${RUN_DIR}/${model_name}/${precision}_tp${tp}"
        mkdir -p "${combo_dir}"
        if [[ "${RESUME_MODE}" == "true" ]] && all_benchmarks_completed_for_combo "${model_path}" "${precision}" "${tp}"; then
          log "Skipping server start for completed combo ${model_name}/${precision}_tp${tp}"
          continue
        fi
        start_server_for_combo "${model_path}" "${precision}" "${tp}" "${model_extra}" "${combo_dir}"

        local concurrency input_len output_len rep
        for concurrency in "${CONCURRENCIES[@]}"; do
          for input_len in "${INPUT_LENS[@]}"; do
            for output_len in "${OUTPUT_LENS[@]}"; do
              for rep in $(seq 1 "${REPEATS}"); do
                if ! run_benchmark_once "${model_path}" "${precision}" "${tp}" "${concurrency}" "${input_len}" "${output_len}" "${rep}" "${combo_dir}"; then
                  log "Benchmark point failed: model=${model_path} precision=${precision} tp=${tp} c=${concurrency} in=${input_len} out=${output_len} rep=${rep}"
                fi
              done
            done
          done
        done

        stop_server
      done
    done
  done

  stop_instance
  print_progress
  log "All requested benchmarks completed."
}

run_through_salloc() {
  local script_abs config_abs workdir inner_cmd
  local -a salloc_cmd script_cmd extra_args=()

  command -v salloc >/dev/null 2>&1 || fail "salloc not found on this machine. Run this script from the AAC login node (for example after: ssh tarun_mishra_qle@aac14.amd.com)."
  command -v srun >/dev/null 2>&1 || fail "srun not found on this machine. Run this script from the AAC login node."

  script_abs="$(readlink -f "$0")"
  workdir="${PWD}"
  if [[ -n "${CONFIG_FILE}" ]]; then
    config_abs="$(readlink -f "${CONFIG_FILE}")"
  else
    config_abs=""
  fi

  inner_cmd="cd $(printf '%q' "${workdir}") && $(printf '%q' "${script_abs}") --inside-allocation"
  if [[ -n "${config_abs}" ]]; then
    inner_cmd+=" -c $(printf '%q' "${config_abs}")"
  fi
  if [[ "${RESUME_MODE}" == "true" ]]; then
    if [[ -n "${RESUME_DIR}" && "${RESUME_DIR}" != "__auto__" ]]; then
      inner_cmd+=" --resume $(printf '%q' "$(readlink -f "${RESUME_DIR}")")"
    else
      inner_cmd+=" --resume"
    fi
  fi
  if [[ "${ASSUME_YES}" == "true" ]]; then
    inner_cmd+=" --yes"
  fi
  script_cmd=(bash -lc "${inner_cmd}")

  salloc_cmd=(salloc
    --job-name "${SLURM_JOB_NAME}"
    --partition "${SLURM_PARTITION}"
    --exclusive
    --mem "${SLURM_MEM}"
    --gres "gpu:${SLURM_GPUS}"
    --time "${SLURM_TIME}"
  )
  if [[ -n "${SLURM_RESERVATION}" ]]; then
    salloc_cmd+=(--reservation "${SLURM_RESERVATION}")
  fi
  if [[ -n "${SLURM_ACCOUNT}" ]]; then
    salloc_cmd+=(--account "${SLURM_ACCOUNT}")
  fi
  if [[ -n "${SLURM_QOS}" ]]; then
    salloc_cmd+=(--qos "${SLURM_QOS}")
  fi
  if [[ -n "${SLURM_EXTRA_ARGS}" ]]; then
    read -ra extra_args <<< "${SLURM_EXTRA_ARGS}"
    salloc_cmd+=("${extra_args[@]}")
  fi
  salloc_cmd+=(
    srun
    --ntasks 1
    --nodes 1
    "${script_cmd[@]}"
  )

  log "Requesting Slurm allocation from login node"
  printf '  %s\n' "$(shell_join "${salloc_cmd[@]}")"
  exec "${salloc_cmd[@]}"
}

confirm_run() {
  local answer
  cat <<EOF
Configuration summary
  Slurm partition : ${SLURM_PARTITION}
  Slurm gpus      : ${SLURM_GPUS}
  Slurm time      : ${SLURM_TIME}
  Reservation     : ${SLURM_RESERVATION:-<none>}
  Container image : ${CONTAINER_IMAGE}
  Models          : ${#MODEL_CONFIGS[@]}
  Base out dir    : ${BASE_OUT_DIR}
EOF
  if [[ "${ASSUME_YES}" == "true" ]]; then
    return
  fi
  read -r -p "Proceed? [y/N] " answer
  [[ "${answer:-n}" =~ ^[Yy]$ ]] || exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      CONFIG_FILE="${2:-}"
      [[ -n "${CONFIG_FILE}" ]] || fail "--config requires a file."
      shift 2
      ;;
    -r|--resume)
      RESUME_MODE=true
      if [[ -n "${2:-}" && "${2:-}" != -* ]]; then
        RESUME_DIR="${2}"
        shift 2
      else
        RESUME_DIR="__auto__"
        shift
      fi
      ;;
    -g|--generate-config)
      GENERATE_CONFIG=true
      if [[ -n "${2:-}" && "${2:-}" != -* ]]; then
        CONFIG_FILE="${2}"
        shift
      fi
      shift
      ;;
    -y|--yes)
      ASSUME_YES=true
      shift
      ;;
    --inside-allocation)
      INSIDE_ALLOCATION=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

if [[ "${GENERATE_CONFIG}" == "true" ]]; then
  generate_sample_config "${CONFIG_FILE:-vllm_runs_slurm.conf}"
  exit 0
fi

if [[ -z "${CONFIG_FILE}" ]]; then
  fail "A config file is required. Use --generate-config to create one."
fi
[[ -f "${CONFIG_FILE}" ]] || fail "Config file not found: ${CONFIG_FILE}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

if [[ "${INSIDE_ALLOCATION}" == "true" || -n "${SLURM_JOB_ID:-}" ]]; then
  confirm_run
  run_inside_allocation
else
  confirm_run
  ASSUME_YES=true
  run_through_salloc
fi
