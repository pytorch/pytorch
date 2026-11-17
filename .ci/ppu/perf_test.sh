#!/usr/bin/env bash
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
echo "[perf] Source dir: $(pwd)"
source .ci/ppu/sdk_env.sh
source .ci/ppu/nccl_env.sh
bash .ci/ppu/install_wheel.sh
bash .ci/ppu/install_triton.sh

echo "=== Env check (CUDA) ==="
echo "pr=${PR_NUMBER:-none} hostname=$(hostname) node=${NODE_NAME:-unknown}"
echo "rank=${RANK:-0} nproc_per_node=${NPROC_PER_NODE:-1}"
python --version
(cd /tmp && python -c "import torch; print('torch', torch.__version__, torch.__file__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())")
ppu-smi || echo "[warn] ppu-smi unavailable; check that the pod has PPU devices"
bash .ci/ppu/install_test_deps.sh
source .ci/ppu/cuda_only_filter.sh

TEST_REPORTS_DIR="$REPO_ROOT/test/test-reports"
mkdir -p "$TEST_REPORTS_DIR"
echo "=== [1/2] CUDA inductor perf tests (run_test.py --include allowlist + -k excluding FP8/CPU) ==="
K_FP8="not fp8 and not float8 and not e4m3 and not e5m2"
K_SKIP_CASES="not test_fusion_choice4_cpu \
and not test_cat \
and not test_cat_pointwise_many_complex_inputs \
and not test_cat_pointwise_many_simple_inputs \
and not test_cat_pointwise_config_option \
and not test_cat_pointwise \
and not test_partitioning_with_view \
and not test_equivalent_template_code \
and not test_split_scan"
python test/run_test.py \
    --include \
        inductor/test_perf \
        inductor/test_benchmark_fusion \
        inductor/test_benchmarking \
        inductor/test_analysis \
    -k "$(ppu_cuda_only_k_expr "$K_FP8" "$K_SKIP_CASES")" \
    --verbose

if [[ "${RUN_GPT_FAST:-1}" != "1" ]]; then
    echo "=== [2/2] gpt_fast microbenchmark: disabled by RUN_GPT_FAST=0, skipping ==="
    echo "[perf] Done"
    exit 0
fi

echo "=== [2/2] gpt_fast microbenchmark (torch.compile end-to-end timing) ==="
if ! (cd /tmp && python -c "import torchao" >/dev/null 2>&1); then
    echo "[perf] torchao unavailable, trying candidate pip indexes (--no-deps)"
    source "$REPO_ROOT/.ci/ppu/pip_sources.sh"
    ppu_build_pip_candidates
    for index in "${PIP_CANDIDATES[@]}" __pip_default__; do
        pip_args=(--disable-pip-version-check --retries 1 --timeout 20 --no-deps)
        if [[ "${index}" == "__pip_default__" ]]; then
            label="pip default index"
        else
            label="${index}"
            pip_args+=(-i "${index}")
        fi
        echo "[perf] Trying index: ${label}"
        if python -m pip install "${pip_args[@]}" torchao; then
            echo "[perf] torchao installed (index: ${label})"
            break
        fi
        echo "[perf][warn] Index ${label} failed, trying next"
    done
fi

if ! (cd /tmp && python -c "import torchao; print('torchao', torchao.__version__)"); then
    cat <<'EOF'
[perf][warn] torchao still unavailable, skipping gpt_fast microbenchmark (does not affect this job's result).
  Why: benchmarks/gpt_fast/benchmark.py -> generate.py does `import torchao` at the top,
       so it cannot even import without torchao; the official install builds from github.com
       source, and the pod has no internet egress.
  Fix: preinstall torchao in the PPU base image (see .github/ci_commit_pins/torchao.txt for the version),
       or make sure the internal pip index has a torchao wheel ABI-compatible with this torch.
EOF
    echo "[perf] Done (part 1 only)"
    exit 0
fi

GPT_FAST_CSV="$TEST_REPORTS_DIR/gpt_fast_benchmark.csv"
rm -f "$GPT_FAST_CSV" "${GPT_FAST_CSV%.csv}.json"
for experiment in mlp_layer_norm_gelu layer_norm gather_gemv gemv; do
    echo "--- gpt_fast: ${experiment} ---"
    python benchmarks/gpt_fast/benchmark.py \
        --only "${experiment}" \
        --output "$GPT_FAST_CSV"
done

echo "=== gpt_fast microbenchmark results ==="
if [[ -s "$GPT_FAST_CSV" ]]; then
    cat "$GPT_FAST_CSV"
else
    echo "[perf][warn] No results in ${GPT_FAST_CSV}: all 4 runs above exited 0 but wrote no csv."
    echo "[perf][warn] Check whether upstream changed --output / output_csv semantics in benchmarks/gpt_fast/benchmark.py."
fi

echo "[perf] Done"
