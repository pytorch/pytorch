#!/usr/bin/env bash
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"

BENCH_CONFIG="${BENCH_CONFIG:-}"
case "$BENCH_CONFIG" in
    huggingface | timm_models | cachebench) ;;
    *)
        echo "[model-perf][error] BENCH_CONFIG must be huggingface / timm_models / cachebench, got: '${BENCH_CONFIG}'" >&2
        exit 1
        ;;
esac

SHARD_NUMBER="${SHARD_NUMBER:-1}"
NUM_TEST_SHARDS="${NUM_TEST_SHARDS:-1}"
if (( SHARD_NUMBER < 1 || SHARD_NUMBER > NUM_TEST_SHARDS )); then
    echo "[model-perf][error] SHARD_NUMBER=${SHARD_NUMBER} not in 1..${NUM_TEST_SHARDS}" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
echo "[model-perf] Source dir: $(pwd)"
echo "[model-perf] Config: BENCH_CONFIG=${BENCH_CONFIG} shard=${SHARD_NUMBER}/${NUM_TEST_SHARDS}"

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
ensure_bench_deps() {
    local want="${BENCH_PIP_PACKAGES:-numpy==1.26.2 scipy==1.14.1 pandas==2.2.3 tqdm>=4.66.0 transformers==5.17.0 timm psutil}"
    [[ -n "$want" ]] || return 0
    local -a specs
    read -r -a specs <<<"$want"
    echo "[model-perf] Installing benchmark deps (pinned): ${specs[*]}"
    local torch_before
    torch_before="$(cd /tmp && python -c 'import torch;print(torch.__version__)')"
    local constraints="/tmp/model_perf_constraints.txt"
    echo "torch==${torch_before}" >"$constraints"
    source "$REPO_ROOT/.ci/ppu/pip_sources.sh"
    ppu_build_pip_candidates
    local index label installed=0
    local pip_args
    for index in "${PIP_CANDIDATES[@]}" __pip_default__; do
        pip_args=(--disable-pip-version-check --retries 1 --timeout 30 -c "$constraints")
        if [[ "${index}" == "__pip_default__" ]]; then
            label="pip default index"
        else
            label="${index}"
            pip_args+=(-i "${index}")
        fi
        echo "[model-perf] Trying index: ${label}"
        if python -m pip install "${pip_args[@]}" "${specs[@]}"; then
            installed=1
            break
        fi
        echo "[model-perf][warn] Index ${label} failed, trying next"
    done
    if [[ "$installed" != "1" ]]; then
        echo "[model-perf][error] No candidate index could install: ${specs[*]}" >&2
        exit 1
    fi
    local torch_after
    torch_after="$(cd /tmp && python -c 'import torch;print(torch.__version__)')"
    if [[ "$torch_before" != "$torch_after" ]]; then
        echo "[model-perf][error] torch changed from ${torch_before} to ${torch_after} after installing benchmark deps; the tested whl was overwritten" >&2
        exit 1
    fi
    echo "[model-perf] Installed versions:"
    python -m pip freeze 2>/dev/null | grep -iE '^(numpy|scipy|pandas|tqdm|transformers|timm|psutil)=' || true
}
ensure_bench_deps

if [[ -n "${HF_HOME:-}" ]]; then
    export HF_HOME
    echo "[model-perf] HF_HOME=${HF_HOME}"
fi
if [[ -n "${TORCHBENCHPATH:-}" ]]; then
    export TORCHBENCHPATH
    export PYTHONPATH="${TORCHBENCHPATH}${PYTHONPATH:+:${PYTHONPATH}}"
    echo "[model-perf] TORCHBENCHPATH=${TORCHBENCHPATH} PYTHONPATH=${PYTHONPATH}"
fi

TEST_REPORTS_DIR="$REPO_ROOT/test/test-reports"
mkdir -p "$TEST_REPORTS_DIR"

PARTITION_FLAGS=()
if (( NUM_TEST_SHARDS > 1 )); then
    PARTITION_FLAGS=(--total-partitions "$NUM_TEST_SHARDS" --partition-id "$((SHARD_NUMBER - 1))")
fi

EXTRA_FLAGS=()
if [[ -n "${MODEL_PERF_EXTRA_ARGS:-}" ]]; then
    read -r -a EXTRA_FLAGS <<<"${MODEL_PERF_EXTRA_ARGS}"
fi

BAD_RESULTS=()
result_ok() {
    local f="$1"
    [[ -s "$f" ]] || return 1
    case "$f" in
        *.csv)
            local rows
            rows="$(grep -cve '^[[:space:]]*$' "$f" || true)"
            (( rows > 1 ))
            ;;
        *.json)
            python - "$f" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as fh:
        data = json.load(fh)
except Exception:
    sys.exit(1)
# null / {} / [] means "no measurements"
sys.exit(0 if data else 1)
PY
            ;;
        *)
            return 0
            ;;
    esac
}

dump_result() {
    local f="$1"
    echo "=== model-perf result: $(basename "$f") ==="
    if [[ -s "$f" ]]; then
        cat "$f"
    fi
    if result_ok "$f"; then
        return 0
    fi
    echo "[model-perf][error] ${f} has no valid measurements (missing/empty/no data rows)." \
         "The benchmark may exit 0 while per-model errors are swallowed; treating as failure." >&2
    BAD_RESULTS+=("$f")
}

case "$BENCH_CONFIG" in
    huggingface)
        out="$TEST_REPORTS_DIR/inductor_huggingface_perf.csv"
        echo "=== CUDA dynamo benchmark: huggingface (inference/inductor/performance) ==="
        python benchmarks/dynamo/huggingface.py \
            --inference --inductor --performance --device cuda \
            --exclude "/" \
            --exclude-exact AllenaiLongformerBase \
            --exclude-exact T5Small \
            --exclude-exact DistillGPT2 \
            --exclude-exact GoogleFnet \
            --exclude-exact YituTechConvBert \
            "${PARTITION_FLAGS[@]}" "${EXTRA_FLAGS[@]}" \
            --output "$out"
        dump_result "$out"
        ;;
    timm_models)
        out="$TEST_REPORTS_DIR/inductor_timm_perf.csv"
        echo "=== CUDA dynamo benchmark: timm_models (inference/inductor/performance) ==="
        python benchmarks/dynamo/timm_models.py \
            --inference --inductor --performance --device cuda \
            "${PARTITION_FLAGS[@]}" "${EXTRA_FLAGS[@]}" \
            --output "$out"
        dump_result "$out"
        ;;
    cachebench)
        echo "=== CUDA dynamo cachebench (torchbench + huggingface) ==="
        for benchmark in torchbench huggingface; do
            for mode in training inference; do
                out="$TEST_REPORTS_DIR/cachebench_${benchmark}_${mode}.json"
                python benchmarks/dynamo/cachebench.py \
                    --mode "$mode" --device cuda --benchmark "$benchmark" --repeat 3 \
                    "${EXTRA_FLAGS[@]}" --output "$out"
                dump_result "$out"
                out_dyn="$TEST_REPORTS_DIR/cachebench_${benchmark}_${mode}_dynamic.json"
                python benchmarks/dynamo/cachebench.py \
                    --mode "$mode" --dynamic --device cuda --benchmark "$benchmark" --repeat 3 \
                    "${EXTRA_FLAGS[@]}" --output "$out_dyn"
                dump_result "$out_dyn"
            done
        done
        ;;
esac

if (( ${#BAD_RESULTS[@]} > 0 )); then
    echo "[model-perf][error] ${#BAD_RESULTS[@]} results have no valid measurements; failing the job:" >&2
    for f in "${BAD_RESULTS[@]}"; do
        echo "  - $f" >&2
    done
    exit 1
fi

echo "[model-perf] Done (config=${BENCH_CONFIG} shard=${SHARD_NUMBER}/${NUM_TEST_SHARDS})"
