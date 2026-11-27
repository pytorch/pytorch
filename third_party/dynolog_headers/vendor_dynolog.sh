#!/usr/bin/env bash
# Re-vendor the dynolog ipcfabric headers used by libkineto's open-source CMake
# build. Kineto's CMake build consumes these three header-only files directly.
#
# Usage:
#   vendor_dynolog.sh <commit> [--from <dynolog-checkout>]
#
#   <commit>            dynolog commit to vendor (required).
#   --from <checkout>   Copy headers from an existing dynolog checkout rooted at
#                       <checkout> instead of cloning from GitHub. Use this where
#                       GitHub is unreachable. The caller must ensure the
#                       checkout is at <commit>.

set -euo pipefail

# Destination is the same directory this script lives in.
dest_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

version_file="${dest_dir}/version.txt"
repo_url="https://github.com/facebookincubator/dynolog.git"
rel_headers="dynolog/src/ipcfabric"
headers=(FabricManager.h Endpoint.h Utils.h)

commit=""
from_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)
      from_dir="${2:?--from requires a path}"
      shift 2
      ;;
    *)
      commit="$1"
      shift
      ;;
  esac
done

if [[ -z "${commit}" ]]; then
  echo "error: a dynolog commit hash is required" >&2
  echo "usage: vendor_dynolog.sh <commit> [--from <dynolog-checkout>]" >&2
  exit 1
fi

tmp_dir=""
cleanup() {
  if [[ -n "${tmp_dir}" ]]; then
    rm -rf "${tmp_dir}"
  fi
}
trap cleanup EXIT

if [[ -n "${from_dir}" ]]; then
  src_dir="${from_dir}"
else
  tmp_dir="$(mktemp -d)"
  git clone --filter=blob:none --no-checkout "${repo_url}" "${tmp_dir}"
  git -C "${tmp_dir}" checkout "${commit}"
  src_dir="${tmp_dir}"
fi

mkdir -p "${dest_dir}/${rel_headers}"
for h in "${headers[@]}"; do
  cp "${src_dir}/${rel_headers}/${h}" "${dest_dir}/${rel_headers}/${h}"
done

# The headers are MIT-licensed; carry dynolog's LICENSE alongside them so the
# vendored copy satisfies the MIT notice requirement and the headers' reference
# to the LICENSE in the root of this source tree resolves.
cp "${src_dir}/LICENSE" "${dest_dir}/LICENSE"

printf '%s\n%s\n' "${repo_url}" "${commit}" >"${version_file}"

echo "Vendored ${headers[*]} + LICENSE from dynolog ${commit}"
echo "  source: ${src_dir}/${rel_headers}"
echo "  dest:   ${dest_dir}/${rel_headers}"
echo "  wrote:  ${version_file}"
