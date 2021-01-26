#!/usr/bin/env bash

set -eou pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

get_python_abi_version() {
    local python_version
    python_version=$1
    python_version_nodot=${python_version/./}
    case "${python_version}" in
        2*|3.5)
            echo "ERROR: Python version ${python_version} is unsupported"
            exit 1
            ;;
        3.6|3.7)
            echo "cp${python_version_nodot}-cp${python_version_nodot}m"
            ;;
        *)
            echo "cp${python_version_nodot}-cp${python_version_nodot}"
            ;;
    esac
}

gen_desired_cuda() {
    # Generates desired cuda for temporary compat with builder scripts
    local cuda_version
    cuda_version=${1:-}
    cuda_version_nodot=${cuda_version/./}
    echo "cu${cuda_version_nodot}"
}

gen_desired_python() {
    # Generates desired python for temporary compat with builder scripts
    local python_version
    python_version=$1
    case "${python_version}" in
        2*|3.5)
            echo "ERROR: Python version ${python_version} is unsupported"
            exit 1
            ;;
        3.6|3.7)
            echo "${python_version}mu"
            ;;
        *)
            echo "${python_version}m"
            ;;
    esac
}


TARGET=${TARGET:-cpu}
TARGET_VERSION=${TARGET_VERSION:-}
MANYLINUX_VERSION=${MANYLINUX_VERSION:-2014}
PYTHON_VERSION=${PYTHON_VERSION:-3.8}
PYTHON_ABI_VERSION=$(get_python_abi_version "${PYTHON_VERSION}")

DOCKER_REGISTRY=${DOCKER_REGISTRY:-docker.io}
DOCKER_ORG=${DOCKER_ORG:-pytorch}
DOCKER_TAG_SUFFIX=${DOCKER_TAG_SUFFIX:-}
DOCKER_IMAGE=${DOCKER_REGISTRY}/${DOCKER_ORG}/binary-manywheel:${TARGET}${TARGET_VERSION/-/}-python${PYTHON_VERSION}${DOCKER_TAG_SUFFIX}

if (set -x && docker manifest inspect ${DOCKER_IMAGE}); then
    echo "Docker image ${DOCKER_IMAGE}, found, pulling..."
    (
        set -x
        docker pull ${DOCKER_IMAGE}
        exit 0
    )
fi

# For CI to set as PROGRESS_FLAG=--progress=plain
PROGRESS_FLAG=${PROGRESS_FLAG:-}

(
    set -x
    DOCKER_BUILDKIT=1 docker build \
        ${PROGRESS_FLAG} \
        --build-arg "DESIRED_PYTHON=$(gen_desired_python ${PYTHON_VERSION})" \
        --build-arg "DESIRED_CUDA=$(gen_desired_cuda ${TARGET_VERSION})" \
        --build-arg "MANYLINUX_VERSION=${MANYLINUX_VERSION}" \
        --build-arg "PYTHON_ABI_VERSION=${PYTHON_ABI_VERSION}" \
        --build-arg "TARGET_VERSION=${TARGET_VERSION}" \
        -t "${DOCKER_IMAGE}" \
        -f "${DIR}/Dockerfile" \
        --target "${TARGET}" \
        "${DIR}/../"
)
