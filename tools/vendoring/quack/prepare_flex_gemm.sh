#!/usr/bin/env bash
# Reconstruct the external QuACK package used by FlexGEMM from a public base.

set -euo pipefail

UPSTREAM_URL="https://github.com/Dao-AILab/quack.git"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
PIN_FILE="$REPO_ROOT/.github/ci_commit_pins/quack.txt"
PATCHES_DIR="$SCRIPT_DIR/flex_gemm_patches"
SERIES_FILE="$PATCHES_DIR/series"

usage() {
    echo "usage: $0 [--src <local-quack-repo>] <empty-destination>" >&2
    exit 2
}

die() {
    echo "prepare_flex_gemm_quack: $*" >&2
    exit 1
}

local_source=""
destination=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --src)
            [[ $# -ge 2 ]] || usage
            local_source=$2
            shift 2
            ;;
        *)
            [[ -z "$destination" ]] || usage
            destination=$1
            shift
            ;;
    esac
done
[[ -n "$destination" ]] || usage

[[ -f "$PIN_FILE" ]] || die "missing pin file: $PIN_FILE"
[[ -f "$SERIES_FILE" ]] || die "missing patch series: $SERIES_FILE"
pinned_sha=$(tr -d '[:space:]' < "$PIN_FILE")
[[ "$pinned_sha" =~ ^[0-9a-f]{40}$ ]] || die "invalid QuACK pin: $pinned_sha"

if [[ -e "$destination" ]]; then
    [[ -d "$destination" && -z "$(ls -A "$destination")" ]] \
        || die "destination must be an empty directory: $destination"
else
    mkdir -p "$destination"
fi

source_url=${local_source:-$UPSTREAM_URL}
if [[ -n "$local_source" ]]; then
    git -C "$local_source" cat-file -e "$pinned_sha^{commit}" 2>/dev/null \
        || die "$local_source does not contain pinned commit $pinned_sha"
fi

git -C "$destination" init --quiet
git -C "$destination" remote add origin "$source_url"
git -C "$destination" fetch --quiet --no-tags --depth=1 origin "$pinned_sha"
git -C "$destination" checkout --quiet --detach FETCH_HEAD

declare -A series_patches=()
patch_count=0
while IFS= read -r patch_name || [[ -n "$patch_name" ]]; do
    [[ -n "$patch_name" && "$patch_name" != \#* ]] || continue
    [[ "$patch_name" != */* && "$patch_name" == *.patch ]] \
        || die "invalid patch-series entry: $patch_name"
    [[ -z "${series_patches[$patch_name]+x}" ]] \
        || die "duplicate patch-series entry: $patch_name"
    patch_file="$PATCHES_DIR/$patch_name"
    [[ -f "$patch_file" ]] || die "patch listed in series not found: $patch_name"
    series_patches[$patch_name]=1
    git -C "$destination" apply --index --unidiff-zero "$patch_file"
    ((patch_count += 1))
done < "$SERIES_FILE"

shopt -s nullglob
for patch_file in "$PATCHES_DIR"/*.patch; do
    patch_name=$(basename "$patch_file")
    [[ -n "${series_patches[$patch_name]+x}" ]] \
        || die "patch file missing from series: $patch_name"
done

git -C "$destination" diff --cached --check
echo "Prepared external QuACK at $pinned_sha with $patch_count FlexGEMM patches"
