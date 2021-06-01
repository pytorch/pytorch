#!/usr/bin/env bash

set -eou pipefail

max() {
    local a=$1
    local b=$2
    if (( $(bc <<< "${a}>${b}") )); then
        echo "${a}"
    else
        echo "${b}"
    fi
}

get_gpu_max_memory_usage_cuda() {
    local my_pid=$1
    local max=$2
    local curr
    # Some processes might not use the GPU
    if ! nvidia-smi pmon -s m -c 1 -o T | grep "${my_pid}" >/dev/null 2>/dev/null; then
        echo "${max}"
        return
    fi
    curr=$(nvidia-smi pmon -s m -c 1 -o T | grep "${my_pid}" | awk '{print $5}' | sort | tail -1 | grep -o "[0-9.]*")
    max "${curr}" "${max}"
}

get_gpu_max_memory_usage_rocm() {
    local my_pid=$1
    local max=$2
    local curr
    # Some processes might not use the GPU
    if ! rocm-smi --showpidusedmem | grep "${my_pid}" >/dev/null 2>/dev/null; then
        echo "${max}"
        return
    fi
    curr=$(rocm-smi --showpidusedmem | grep "${my_pid}" | awk '{print $3}' | sort | tail -1 |  grep -o "[0-9.]*")
    max "$(($curr/1000000))" "${max}"
}

get_cpu_max_rss_memory_usage() {
    local my_pid=$1
    local max=$2
    local curr
    local SMAPS_FILE="/proc/${my_pid}/smaps"
    if [[ -f "${SMAPS_FILE}" ]]; then
        curr=$(cat "${SMAPS_FILE}" | grep -i '^Rss' |  awk '{Total+=$2} END {print Total/1024""}' | grep -o "[0-9.]*")
        max "${curr}" "${max}"
    else
        echo "${max}"
    fi
}

get_cpu_max_pss_memory_usage() {
    local my_pid=$1
    local max=$2
    local curr
    local SMAPS_FILE="/proc/${my_pid}/smaps"
    if [[ -f "${SMAPS_FILE}" ]]; then
        curr=$(cat "${SMAPS_FILE}" | grep -i '^Pss' |  awk '{Total+=$2} END {print Total/1024""}' | grep -o "[0-9.]*")
        max "${curr}" "${max}"
    else
        echo "${max}"
    fi
}

get_multi_proc_ids() {
    local my_pid=$1
    # If a process spawns other processes this should catch that
    ps --forest -o pid --ppid "${my_pid}" --pid "${my_pid}" | awk 'NR>1 {print}'
}

LOG_FILE=${LOG_FILE:-}
if [[ -z "${LOG_FILE}" ]]; then
    LOG_FILE=$(mktemp --suffix "_monitor_proc")
fi

MEM_FILE=${MEM_FILE:-}
if [[ -z "${MEM_FILE}" ]]; then
    MEM_FILE=${LOG_FILE}_mem
fi

COMMAND=$@
if [[ "${COMMAND^^}" == *"ROCM"* ]]; then
    get_gpu_max_memory_usage="get_gpu_max_memory_usage_rocm"
    prefix=$1
    COMMAND_TO_RUN=${COMMAND#"$prefix"}
else
    get_gpu_max_memory_usage="get_gpu_max_memory_usage_cuda"
    COMMAND_TO_RUN=$@
fi

echo "Running  ${COMMAND_TO_RUN} > \"${LOG_FILE}\" 2>&1 &"
/usr/bin/time -f "\nTotal time elapsed: %e seconds." ${COMMAND_TO_RUN} > "${LOG_FILE}" 2>&1 &
PID_TO_WATCH=$(ps aux |  grep -v 'grep' | grep -F "${COMMAND_TO_RUN}" | grep -v "$0" | awk '{print $2}' | head -1)
trap "kill -9 ${PID_TO_WATCH} >/dev/null 2>/dev/null || exit 0" $(seq 0 15)
MAX_GPU_MEMORY=0
MAX_RSS_MEMORY=0
MAX_PSS_MEMORY=0

echo "Watching PID: ${PID_TO_WATCH}"
echo "Searching for spawned processes with"
echo "Tail log file with: ps --forest -o pid --ppid ${PID_TO_WATCH} --pid ${PID_TO_WATCH} | awk 'NR>1 {print}'"
echo "    tail -f ${LOG_FILE}"

iteration=0
printf "\n%-15s%-15s%s\n" "Max GPU Mem." "Max RSS Mem." "Max PSS Mem." | tee $MEM_FILE
while kill -0 "${PID_TO_WATCH}" >/dev/null 2>/dev/null; do
    for pid in $(get_multi_proc_ids "${PID_TO_WATCH}"); do
        MAX_GPU_MEMORY=$("${get_gpu_max_memory_usage}" "${pid}" "${MAX_GPU_MEMORY}")
        MAX_RSS_MEMORY=$(get_cpu_max_rss_memory_usage "${pid}" "${MAX_RSS_MEMORY}")
        MAX_PSS_MEMORY=$(get_cpu_max_pss_memory_usage "${pid}" "${MAX_PSS_MEMORY}")
    done
    # Print out max every 5 seconds
    if [[ "${iteration}" -eq "10" ]]; then
        printf "%-15s%-15s%s\n" "${MAX_GPU_MEMORY}" "${MAX_RSS_MEMORY}" "${MAX_PSS_MEMORY}" | tee -a $MEM_FILE
        iteration=0
    else
        iteration=$((iteration+1))
    fi
    sleep 0.5
done

printf "%-15s%-15s%s\n" "${MAX_GPU_MEMORY}" "${MAX_RSS_MEMORY}" "${MAX_PSS_MEMORY}" | tee -a $MEM_FILE
