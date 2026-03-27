#!/bin/bash
# RE equivalent of pytorch/test-infra/.github/actions/setup-uv
set -eu
: "${PYTHON_VERSION:=3.12}"
: "${UV_VERSION:=0.9.21}"

if ! command -v uv &>/dev/null; then
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv python install "$PYTHON_VERSION"
uv venv --seed --python "$PYTHON_VERSION"
source .venv/bin/activate
