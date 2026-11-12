#!/bin/bash
# Usage: fetch-submodules.sh [FILTER] [DEPTH]
# Recursively checks out the current repo's submodules, optionally with a partial
# clone FILTER (e.g. tree:0) or a shallow DEPTH. git's --recursive visits one
# submodule at a time, so each top-level submodule's nested tree (where the
# copies of cutlass and composable_kernel live) is fetched in parallel.
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"

args=(--init --force --jobs 8)
[[ -n "${1:-}" ]] && args+=("--filter=$1")
[[ "${2:-0}" -gt 0 ]] && args+=("--depth=$2")

fetch() {
  git submodule update "${args[@]}" &&
    git submodule foreach --quiet 'echo "$sm_path"' |
    xargs -P 8 -I{} git -C {} submodule update --recursive "${args[@]}"
}
# Already-fetched submodules make a retry cheap.
fetch || (sleep 4 && fetch) || (sleep 16 && fetch)
