#!/usr/bin/env bash
export PYTORCH_TESTING_DEVICE_ONLY_FOR="cuda"
_ppu_cuda_only_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! PPU_K_CPU_ONLY="$(python "${_ppu_cuda_only_dir}/list_cpu_only_test_classes.py" \
    "${_ppu_cuda_only_dir}/../../test")"; then
    echo "[cuda-only][error] Failed to compute CPU test class exclusion expression (see above)" >&2
    exit 1
fi
unset _ppu_cuda_only_dir

echo "[cuda-only] Layer 1: PYTORCH_TESTING_DEVICE_ONLY_FOR=${PYTORCH_TESTING_DEVICE_ONLY_FOR}"
echo "[cuda-only] Layer 2: -k '${PPU_K_CPU_ONLY}'"

ppu_cuda_only_k_expr() {
    local out="${PPU_K_CPU_ONLY}"
    local part
    for part in "$@"; do
        if [[ -n "${part}" ]]; then
            out="${out} and ${part}"
        fi
    done
    printf '%s' "${out}"
}
