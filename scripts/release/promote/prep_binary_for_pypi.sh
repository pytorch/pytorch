#!/usr/bin/env bash

# Preps binaries for publishing to pypi by removing the
# version suffix we normally add for all binaries
# (outside of default ones, CUDA 10.2 currently)

# Usage is:
# $ prep_binary_for_pypy.sh <path_to_whl_file> <path_to_multiple_whl_files>

# Will output a whl in your current directory

set -eou pipefail
shopt -s globstar

OUTPUT_DIR=${OUTPUT_DIR:-$(pwd)}

tmp_dir="$(mktemp -d)"
trap 'rm -rf ${tmp_dir}' EXIT

for whl_file in "$@"; do
    whl_file=$(realpath "${whl_file}")
    whl_dir="${tmp_dir}/$(basename "${whl_file}")_unzipped"
    mkdir -pv "${whl_dir}"
    (
        set -x
        unzip -q "${whl_file}" -d "${whl_dir}"
    )
    version_with_suffix=$(grep '^Version:' "${whl_dir}"/*/METADATA | cut -d' ' -f2)
    version_with_suffix_escaped=${version_with_suffix/+/%2B}
    # Remove all suffixed +bleh versions
    version_no_suffix=${version_with_suffix/+*/}
    new_whl_file=${OUTPUT_DIR}/$(basename "${whl_file/${version_with_suffix_escaped}/${version_no_suffix}}")
    dist_info_folder=$(find "${whl_dir}" -type d -name '*.dist-info' | head -1)
    basename_dist_info_folder=$(basename "${dist_info_folder}")
    dirname_dist_info_folder=$(dirname "${dist_info_folder}")
    (
        set -x
        find "${dist_info_folder}" -type f -exec sed -i "s!${version_with_suffix}!${version_no_suffix}!" {} \;
        # Moves distinfo from one with a version suffix to one without
        # Example: torch-1.8.0+cpu.dist-info => torch-1.8.0.dist-info
        mv "${dist_info_folder}" "${dirname_dist_info_folder}/${basename_dist_info_folder/${version_with_suffix}/${version_no_suffix}}"
        cd "${whl_dir}"
        zip -qr "${new_whl_file}" .
    )
done
