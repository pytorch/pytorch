#!/usr/bin/env bash

set -eou pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${DIR}/common_utils.sh"


# Allow for users to pass PACKAGE_NAME
# For use with other packages, i.e. torchvision, etc.
PACKAGE_NAME=${PACKAGE_NAME:-pytorch}
PYTORCH_CONDA_FROM=${PYTORCH_CONDA_FROM:-pytorch-test}
PYTORCH_CONDA_TO=${PYTORCH_CONDA_TO:-pytorch}
CONDA_PLATFORMS="linux-64 osx-64 win-64 noarch"

pytorch_version="$(get_pytorch_version)"

tmp_dir="$(mktemp -d)"
pushd "${tmp_dir}"
trap 'rm -rf ${tmp_dir}' EXIT

conda_search() {
    conda search -q "${PYTORCH_CONDA_FROM}::${PACKAGE_NAME}==${pytorch_version}" -c "${PYTORCH_CONDA_FROM}" --platform "${platform}" \
        | grep -e "^${PACKAGE_NAME}" \
        | awk -F ' *' '{print $3}' \
        | xargs -I % echo "https://anaconda.org/${PYTORCH_CONDA_FROM}/${PACKAGE_NAME}/${pytorch_version}/download/${platform}/${PACKAGE_NAME}-${pytorch_version}-%.tar.bz2"
}

pkgs_to_download=()
for platform in ${CONDA_PLATFORMS}; do
    pkgs_to_download+=($(\
        conda_search 2>/dev/null || true
    ))
    # Create directory where packages will eventually be downloaded
    mkdir -p "${platform}"
done

my_curl() {
    local dl_url=$1
    local start=$(date +%s)
    # downloads should be distinguished by platform which should be the second
    # to last field in the url, this is to avoid clobbering same named files
    # for different platforms
    dl_dir=$(echo "${dl_url}" | rev | cut -d'/' -f 2 | rev)
    dl_name=$(echo "${dl_url}" | rev | cut -d'/' -f 1 | rev)
    curl -fsSL -o "${dl_dir}/${dl_name}" "${dl_url}"
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
    # We use --skip here to avoid re-uploading files we've already uploaded
    set -x
    ${ANACONDA} upload --skip -u ${PYTORCH_CONDA_TO} $(find . -name '*.bz2')
)

popd
