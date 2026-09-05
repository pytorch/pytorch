#!/usr/bin/env bash

set -eu -o pipefail

pytorch_cmake_version="4.4.2"
pytorch_cmake_root="${RUNNER_TEMP:-/tmp}/cmake-${pytorch_cmake_version}"
pytorch_cmake_bin="${pytorch_cmake_root}/cmake/data/bin"

if [[ ! -x "${pytorch_cmake_bin}/cmake" || ! -x "${pytorch_cmake_bin}/ctest" ]]; then
    if command -v uv >/dev/null; then
        uv pip install --no-deps --target "${pytorch_cmake_root}" "cmake==${pytorch_cmake_version}"
    elif [[ -x /opt/python/cp310-cp310/bin/python ]]; then
        /opt/python/cp310-cp310/bin/python -m pip install \
            --disable-pip-version-check \
            --no-deps \
            --target "${pytorch_cmake_root}" \
            "cmake==${pytorch_cmake_version}"
    else
        echo "No Python-independent package installer is available for CMake" >&2
        return 1
    fi
fi

export CMAKE_EXECUTABLE="${pytorch_cmake_bin}/cmake"
export PATH="${pytorch_cmake_bin}:${PATH}"

"${CMAKE_EXECUTABLE}" --version
"${pytorch_cmake_bin}/ctest" --version
