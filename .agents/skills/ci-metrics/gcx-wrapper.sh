#!/usr/bin/env bash

set -euo pipefail

export GCX_SERVER="${GCX_SERVER:-https://pytorchci.grafana.net}"
export GCX_CONTEXT="${GCX_CONTEXT:-pytorchci}"
# gcx 1.x stores credentials in the OS keychain and refuses to write them
# without one, which headless machines lack. The token is a read-only Viewer
# token, so keep it in gcx's 0600 config file as gcx 0.x did.
export GCX_KEYCHAIN="${GCX_KEYCHAIN:-off}"

# Cache a pinned, prebuilt gcx binary in a private directory and invoke it by
# absolute path, so nothing lands on the user's PATH. gcx's installer verifies
# the download's SHA-256 checksum.
GCX_VERSION="${GCX_VERSION:-1.7.1}"
GCX_CACHE="${GCX_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/pytorch-ci-metrics}"
GCX_BIN="${GCX_CACHE}/gcx-${GCX_VERSION}"

if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is needed to download gcx and fetch its token." >&2
  exit 1
fi

_ensure_gcx() {
  [[ -x "${GCX_BIN}" ]] && return 0

  echo "Fetching gcx ${GCX_VERSION} to ${GCX_BIN}..." >&2
  mkdir -p "${GCX_CACHE}"
  local tmp
  tmp="$(mktemp -d "${GCX_CACHE}/install.XXXXXX")"
  local installer="https://raw.githubusercontent.com/grafana/gcx/v${GCX_VERSION}/scripts/install.sh"
  if ! curl -fsSL "${installer}" | INSTALL_DIR="${tmp}" VERSION="${GCX_VERSION}" sh >/dev/null; then
    rm -rf "${tmp}"
    echo "error: failed to install gcx ${GCX_VERSION}" >&2
    return 1
  fi
  if ! mv "${tmp}/gcx" "${GCX_BIN}"; then
    rm -rf "${tmp}"
    echo "error: gcx installer did not produce a binary" >&2
    return 1
  fi
  rm -rf "${tmp}"
}

_run_gcx() {
  "${GCX_BIN}" "$@"
}

_login_gcx() {
  echo "Authenticating Grafana..." >&2

  if ! command -v gh >/dev/null 2>&1; then
    echo "error: gh CLI is needed to fetch the gcx token"
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
  if ! gcx_token="$(printf 'Authorization: Bearer %s\n' "${gh_token}" |
    curl -fsSL -H @- "https://hud.pytorch.org/api/gcx-token?token_name=$HOSTNAME" 2>&1)"; then
    echo "error: failed to fetch gcx token from HUD: ${gcx_token}"
    return 1
  fi

  if ! GRAFANA_TOKEN="${gcx_token}" _run_gcx login "${GCX_CONTEXT}" \
    --server "${GCX_SERVER}" \
    --yes >/dev/null 2>&1; then
    echo "error: gcx login failed for ${GCX_CONTEXT}"
    return 1
  fi
}

if ! _ensure_gcx; then
  exit 1
fi

# /api/health answers without credentials; /api/user fails on a revoked token.
if ! _run_gcx --no-color api --context "${GCX_CONTEXT}" /api/user >/dev/null 2>&1; then
  if ! login_error="$(_login_gcx)"; then
    echo "${login_error}" >&2
    exit 1
  fi
fi

_run_gcx config use-context "${GCX_CONTEXT}" >/dev/null 2>&1 || true

exec "${GCX_BIN}" "$@"
