#!/usr/bin/env bash

set -eou pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${DIR}/common_utils.sh"


# Allow for users to pass PACKAGE_NAME
# For use with other packages, i.e. torchvision, etc.
PACKAGE_NAME=${PACKAGE_NAME:-pytorch}
PYTORCH_CONDA_FROM=${PYTORCH_CONDA_FROM:-pytorch-test}
PYTORCH_CONDA_TO=${PYTORCH_CONDA_TO:-pytorch}
CONDA_PLATFORMS="linux-64 osx-64 win-64"

pytorch_version="$(get_pytorch_version)"

tmp_dir="$(mktemp -d)"
pushd "${tmp_dir}"
trap 'rm -rf ${tmp_dir}' EXIT

pkgs_to_download=()
for platform in ${CONDA_PLATFORMS}; do
    pkgs_to_download+=($(\
        conda search "${PYTORCH_CONDA_FROM}::${PACKAGE_NAME}==${pytorch_version}" -c "${PYTORCH_CONDA_FROM}" --platform "${platform}" \
            | grep "${PACKAGE_NAME}" \
            | awk -F ' *' '{print $3}' \
            | xargs -I % echo "https://anaconda.org/${PYTORCH_CONDA_FROM}/${PACKAGE_NAME}/${pytorch_version}/download/${platform}/${PACKAGE_NAME}-${pytorch_version}-%.tar.bz2"
    ))
done

my_curl() {
    local dl_url=$1
    local start=$(date +%s)
    curl -fsSL -O "${dl_url}"
    local end=$(date +%s)
    local diff=$(( end - start ))
    echo "+ ${dl_url} took ${diff}s"
}
export -f my_curl

# Download all packages in parallel
printf '%s\n' "${pkgs_to_download[@]}" \
    | xargs -P 10 -I % bash -c '(declare -t my_curl); my_curl %'

# dry run by default
DRY_RUN=${DRY_RUN:-enabled}
ANACONDA="true anaconda"
if [[ $DRY_RUN = "disabled" ]]; then
    ANACONDA="anaconda"
fi
(
    set -x
    ${ANACONDA} upload -u ${PYTORCH_CONDA_TO} *.bz2
)

popd
