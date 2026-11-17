#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WHEEL_DIR="${WHEEL_DIR:-$REPO_ROOT/.ci/ppu/wheelhouse}"
shopt -s nullglob
WHLS=("$WHEEL_DIR"/torch-*.whl)
if [[ ${#WHLS[@]} -eq 0 ]]; then
    cat >&2 <<EOF
[wheel][error] No torch-*.whl found in $WHEEL_DIR.
Possible causes:
  - the workflow step that downloads the ppu-ci-810/890 build job whl did not run (or saved elsewhere);
  - ppu-distributed-action dropped the whl when packing sources. The dir must be inside the source tree
    and not matched by .gitignore (hence .ci/ppu/wheelhouse instead of dist/, which is gitignored
    and would be dropped entirely by gitignore-filtered packing).
Current dir contents:
EOF
    ls -la "$WHEEL_DIR" >&2 2>/dev/null || echo "  (dir does not exist)" >&2
    exit 1
fi
if [[ ${#WHLS[@]} -gt 1 ]]; then
    echo "[wheel][error] Multiple torch whls in $WHEEL_DIR, cannot pick one:" >&2
    printf '  %s\n' "${WHLS[@]}" >&2
    exit 1
fi
WHEEL="${WHLS[0]}"
echo "=== Installing PPU torch whl ==="
echo "[wheel] Package: $(basename "$WHEEL") ($(du -h "$WHEEL" | cut -f1))"

if python -m pip uninstall -y torch >/dev/null 2>&1; then
    echo "[wheel] Uninstalled existing torch"
else
    echo "[wheel] No torch preinstalled (expected for the PPU base image)"
fi

source "$REPO_ROOT/.ci/ppu/pip_sources.sh"
ppu_build_pip_candidates

installed=0
for index in "${PIP_CANDIDATES[@]}" __pip_default__; do
    pip_args=(--disable-pip-version-check --no-cache-dir --retries 1 --timeout 20)
    if [[ "${index}" == "__pip_default__" ]]; then
        label="pip default index"
    else
        label="${index}"
        pip_args+=(-i "${index}")
    fi
    echo "[wheel] Trying index: ${label}"
    if python -m pip install "${pip_args[@]}" "$WHEEL"; then
        echo "[wheel] Installed (index: ${label})"
        installed=1
        break
    fi
    echo "[wheel][warn] Index ${label} failed, trying next"
done

if [[ "${installed}" -ne 1 ]]; then
    cat >&2 <<'EOF'
[wheel][error] Failed to install the whl. torch itself is a local file, so the failure is almost
always its runtime deps (filelock / typing-extensions / setuptools / sympy / networkx / jinja2 /
fsspec) being unavailable, or a full disk. See the "pip source connectivity check" section
of install_test_deps.sh for how to diagnose pip indexes.
EOF
    exit 1
fi

(
    cd /tmp
    python - "$REPO_ROOT" <<'PY'
import sys

import torch

repo_root = sys.argv[1]
print("torch", torch.__version__, torch.__file__)
print("git_version", getattr(torch.version, "git_version", "<unknown>"))
print("built_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())

if torch.__file__.startswith(repo_root):
    sys.exit(f"[wheel] Imported torch from the source tree ({torch.__file__}); whl not in effect")
PY
)

echo "[wheel] Done"
