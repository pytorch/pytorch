#!/usr/bin/env bash

set -eou pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${DIR}/common_utils.sh"

# Allow for users to pass PACKAGE_NAME
# For use with other packages, i.e. torchvision, etc.
PACKAGE_NAME=${PACKAGE_NAME:-torch}

pytorch_version="$(get_pytorch_version)"
# Refers to the specific package we'd like to promote
# i.e. VERSION_SUFFIX='%2Bcu102'
#      torch-1.8.0+cu102 -> torch-1.8.0
VERSION_SUFFIX=${VERSION_SUFFIX:-}
# Refers to the specific platofmr we'd like to promote
# i.e. PLATFORM=linux_x86_64
# For domains like torchaudio / torchtext this is to be left blank
PLATFORM=${PLATFORM:-}

pkgs_to_promote=$(\
    curl -fsSL https://download.pytorch.org/whl/torch_stable.html \
        | grep "${PACKAGE_NAME}-${pytorch_version}${VERSION_SUFFIX}-" \
        | grep "${PLATFORM}" \
        | cut -d '"' -f2
)

tmp_dir="$(mktemp -d)"
output_tmp_dir="$(mktemp -d)"
trap 'rm -rf ${tmp_dir} ${output_tmp_dir}' EXIT
pushd "${output_tmp_dir}"

# Dry run by default
DRY_RUN=${DRY_RUN:-enabled}
# On dry run just echo the commands that are meant to be run
TWINE_UPLOAD="echo twine upload"
if [[ $DRY_RUN = "disabled" ]]; then
    TWINE_UPLOAD="twine upload"
fi

for pkg in ${pkgs_to_promote}; do
    pkg_basename="$(basename "${pkg}")"
    # Don't attempt to change if manylinux2014
    if [[ "${pkg}" != *manylinux2014* ]]; then
        pkg_basename="$(basename "${pkg//linux/manylinux1}")"
    fi
    orig_pkg="${tmp_dir}/${pkg_basename}"
    (
        set -x
        # Download package, sub out linux for manylinux1
        curl -fsSL -o "${orig_pkg}" "https://download.pytorch.org/whl/${pkg}"
    )

    if [[ -n "${VERSION_SUFFIX}" ]]; then
        OUTPUT_DIR="${output_tmp_dir}" ${DIR}/prep_binary_for_pypi.sh "${orig_pkg}"
    else
        mv "${orig_pkg}" "${output_tmp_dir}/"
    fi

    (
        set -x
        ${TWINE_UPLOAD} \
            --disable-progress-bar \
            --non-interactive \
            ./*.whl
        rm -rf ./*.whl
    )
done
