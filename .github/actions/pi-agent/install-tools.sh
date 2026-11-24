#!/usr/bin/env bash
# Pinned native dependencies for pi's offline grep/find tools on Linux x86_64.
set -euo pipefail

if [[ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]]; then
  echo "::error::pi search-tool installation currently supports Linux x86_64 only"
  exit 1
fi

bin_dir="${RUNNER_TEMP:?}/pi-tools/bin"
work=$(mktemp -d "$RUNNER_TEMP/pi-tools-download.XXXXXX")
trap 'rm -rf "$work"' EXIT
mkdir -p "$bin_dir"

install_tool() {
  local binary=$1 repo=$2 tag=$3 archive_root=$4 checksum=$5
  local archive="$work/$binary.tar.gz"
  curl --fail --location --silent --show-error --proto '=https' --proto-redir '=https' \
    --retry 3 --connect-timeout 10 --max-time 120 \
    "https://github.com/$repo/releases/download/$tag/$archive_root.tar.gz" -o "$archive"
  if ! printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check --status; then
    echo "::error::SHA-256 mismatch for $binary; refusing to extract or execute"
    exit 1
  fi
  # Extract only the verified binary, not other release archive contents.
  tar -xzf "$archive" -C "$work" "$archive_root/$binary"
  install -m 0755 "$work/$archive_root/$binary" "$bin_dir/$binary"
  "$bin_dir/$binary" --version
}

# SHA-256 values verified against each release asset's GitHub digest and downloaded bytes.
install_tool rg BurntSushi/ripgrep 15.2.0 ripgrep-15.2.0-x86_64-unknown-linux-musl \
  33e15bcf1624b25cdd2a55813a47a2f95dbe126268203e76aa6a585d1e7b149c
install_tool fd sharkdp/fd v10.5.0 fd-v10.5.0-x86_64-unknown-linux-musl \
  761c72dc8e120d85b22292063be8a796e2eeb20eb3e4f38b8fa2343ccf3514a7

printf '%s\n' "$bin_dir" >> "${GITHUB_PATH:?}"
