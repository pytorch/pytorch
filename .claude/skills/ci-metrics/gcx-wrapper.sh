#!/usr/bin/env bash

set -euo pipefail

export GCX_SERVER="${GCX_SERVER:-https://pytorchci.grafana.net}"
export GCX_CONTEXT="${GCX_CONTEXT:-pytorchci}"
export GCX_NO_UPDATE_NOTIFIER="${GCX_NO_UPDATE_NOTIFIER:-1}"
GCX_CMD=(go run "${GCX_MODULE:-github.com/grafana/gcx/cmd/gcx@latest}")

if ! command -v go >/dev/null 2>&1; then
  echo "error: go is needed to run gcx." >&2
  exit 1
fi

_run_gcx() {
  "${GCX_CMD[@]}" "$@"
}

_login_gcx() {
  echo "Authenticating Grafana..." >&2

  if ! command -v gh >/dev/null 2>&1; then
    echo "error: gh CLI is needed to fetch the gcx token"
    return 1
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo "error: curl is needed to fetch the gcx token"
    return 1
  fi

  if ! gh auth status >/dev/null 2>&1; then
    echo "error: gh is not authorized. Run 'gh auth login --hostname github.com --git-protocol ssh --web' and retry."
    return 1
  fi

  local gh_token
  if ! gh_token="$(gh auth token 2>/dev/null)"; then
    echo "error: failed to read gh auth token"
    return 1
  fi

  local gcx_token
  if ! gcx_token="$(curl -fsSL -H "Authorization: Bearer ${gh_token}" "https://hud.pytorch.org/api/gcx-token?token_name=$HOSTNAME" 2>&1)"; then
    echo "error: failed to fetch gcx token from HUD: ${gcx_token}"
    return 1
  fi

  if ! _run_gcx login "${GCX_CONTEXT}" \
    --server "${GCX_SERVER}" \
    --yes \
    --token "${gcx_token}" >/dev/null 2>&1; then
    echo "error: gcx login failed for ${GCX_CONTEXT}"
    return 1
  fi
}

if ! _run_gcx --no-color api --context "${GCX_CONTEXT}" /api/health >/dev/null 2>&1; then
  if ! login_error="$(_login_gcx)"; then
    echo "${login_error}" >&2
    exit 1
  fi
fi

_run_gcx config use-context "${GCX_CONTEXT}" >/dev/null 2>&1 || true

exec "${GCX_CMD[@]}" "$@"
