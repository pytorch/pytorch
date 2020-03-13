#!/usr/bin/env bash

set -eou pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${DIR}/common_utils.sh"

# Allow for users to pass PACKAGE_NAME
# For use with other packages, i.e. torchvision, etc.
PACKAGE_NAME=${PACKAGE_NAME:-torch}

pytorch_version="$(get_pytorch_version)"

# This assumes you have already promoted the wheels to stable S3
pkgs_to_promote=$(\
    curl -fsSL https://download.pytorch.org/whl/torch_stable.html \
        | grep "${PACKAGE_NAME}-${pytorch_version}" \
        | grep -v "%2B" \
        | grep -v "win_amd64" \
        | cut -d '"' -f2
)

tmp_dir="$(mktemp -d)"
trap 'rm -rf ${tmp_dir}' EXIT
pushd "${tmp_dir}"

# Dry run by default
DRY_RUN=${DRY_RUN:-enabled}
# On dry run just echo the commands that are meant to be run
TWINE_UPLOAD="echo twine upload"
if [[ $DRY_RUN = "disabled" ]]; then
    TWINE_UPLOAD="twine upload"
fi

for pkg in ${pkgs_to_promote}; do
    pkg_basename="$(basename "${pkg//linux/manylinux1}")"
    (
        set -x
        # Download package, sub out linux for manylinux1
        curl -fsSL -o "${pkg_basename}" "https://download.pytorch.org/whl/${pkg}"
        ${TWINE_UPLOAD} \
            --disable-progress-bar \
            --non-interactive \
            "${pkg_basename}"
    )
done
