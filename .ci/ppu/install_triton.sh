#!/usr/bin/env bash
set -euo pipefail

TRITON_VERSION="${TRITON_VERSION:-3.7.0}"
TRITON_INDEX="${TRITON_INDEX:-${PIP_INDEX:-}}"

if [[ -z "${TRITON_INDEX}" ]]; then
    cat >&2 <<'EOF'
[triton][error] Neither TRITON_INDEX nor PIP_INDEX is set; cannot determine the triton source.
triton must come from the internal pypi_index (the same version on public pypi is not the PPU build),
so there is intentionally no default-index fallback. Set TRITON_INDEX or PIP_INDEX in the workflow command.
EOF
    exit 1
fi

echo "=== Installing triton==${TRITON_VERSION} (only index: ${TRITON_INDEX}) ==="
env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE=/dev/null \
    python -m pip install \
    --disable-pip-version-check --no-cache-dir --retries 3 --timeout 60 \
    --index-url "${TRITON_INDEX}" \
    "triton==${TRITON_VERSION}"

(cd /tmp && python -c "import triton; print('triton', triton.__version__, triton.__file__)")

echo "[triton] Done"
