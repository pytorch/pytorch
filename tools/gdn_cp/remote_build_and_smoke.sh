#!/usr/bin/env bash
# Isolated venv + overlay patched torch Python + compile GDN CUDA ext + 2-GPU smoke.
# Must run inside tmux session `torch`. Does not touch train/.venv.
set -euo pipefail

ROOT="${TORCH_WORK_ROOT:-/root/autodl-tmp/torch-work}"
SRC="${ROOT}/pytorch"
VENV="${ROOT}/.venv"
MEM_CAP_MIB="${MEM_CAP_MIB:-500}"
export MEM_CAP_MIB

mkdir -p "${ROOT}"
cd "${ROOT}"

if [[ ! -d "${SRC}/.git" ]]; then
  echo "[gdn] cloning LambdaLinker/pytorch from gitcode"
  git clone --depth 1 --branch gdn-cp "https://gitcode.com/LambdaLinker/pytorch.git" "${SRC}"
else
  echo "[gdn] pulling gdn-cp"
  git -C "${SRC}" fetch --depth 1 origin gdn-cp
  git -C "${SRC}" checkout gdn-cp
  git -C "${SRC}" reset --hard origin/gdn-cp
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -U pip setuptools wheel ninja

if ! python -c "import torch; assert torch.__version__.startswith('2.13.0')" 2>/dev/null; then
  echo "[gdn] installing torch==2.13.0 into isolated venv"
  # Prefer AutoDL / Tsinghua mirrors; cu130 wheel lives on download.pytorch.org.
  python -m pip install torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/cu130 \
    || python -m pip install torch==2.13.0 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

python -c "import torch; print('venv torch', torch.__version__, 'cuda', torch.version.cuda, 'git', torch.version.git_version)"

python "${SRC}/tools/gdn_cp/overlay_into_site_packages.py" --src "${SRC}"

# Compile the GDN CUDA extension without occupying the training GPUs.
echo "[gdn] compiling gated_delta_ext (CPU/nvcc, no GPU occupancy)"
(
  cd "${SRC}/extensions/gated_delta"
  export CUDA_VISIBLE_DEVICES=""
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
  python setup.py install
) || echo "[gdn] CUDA extension build failed; smoke will use Python GDN"

echo "[gdn] nvidia-smi before smoke"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true

echo "[gdn] launching 2-GPU FSDP+CP smoke, ${MEM_CAP_MIB} MiB/GPU cap"
cd "${SRC}"
torchrun --standalone --nproc_per_node=2 tools/gdn_cp/smoke_fsdp_cp.py
