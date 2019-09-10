#!/bin/bash

# shellcheck disable=SC2034
source "$(dirname "${BASH_SOURCE[0]}")/macos-common.sh"

git submodule sync --recursive
git submodule update --init --recursive
export CMAKE_PREFIX_PATH=${WORKSPACE_DIR}/miniconda3/

# Build PyTorch
if [[ "${BUILD_ENVIRONMENT}" == *cuda9.2* ]]; then
  export CUDA_VERSION=9.2
  export TORCH_CUDA_ARCH_LIST=5.2
  export PATH=/Developer/NVIDIA/CUDA-${CUDA_VERSION}/bin${PATH:+:${PATH}}
  export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-${CUDA_VERSION}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}
  export CUDA_HOME=/Developer/NVIDIA/CUDA-${CUDA_VERSION}
  export USE_CUDA=1

  if [ -z "${IN_CIRCLECI}" ]; then
    # Eigen gives "explicit specialization of class must precede its first use" error
    # when compiling with Xcode 9.1 toolchain, so we have to use Xcode 8.2 toolchain instead.
    export DEVELOPER_DIR=/Library/Developer/CommandLineTools
  fi
else
  if [ -z "${IN_CIRCLECI}" ]; then
    export DEVELOPER_DIR=/Applications/Xcode9.app/Contents/Developer
  fi
fi

if which sccache > /dev/null; then
  printf "#!/bin/sh\nexec sccache $(which clang++) \$*" > "${WORKSPACE_DIR}/clang++"
  chmod a+x "${WORKSPACE_DIR}/clang++"

  printf "#!/bin/sh\nexec sccache $(which clang) \$*" > "${WORKSPACE_DIR}/clang"
  chmod a+x "${WORKSPACE_DIR}/clang"

  if [[ "${BUILD_ENVIRONMENT}" == *cuda* ]]; then
    printf "#!/bin/sh\nexec sccache $(which nvcc) \$*" > "${WORKSPACE_DIR}/nvcc"
    chmod a+x "${WORKSPACE_DIR}/nvcc"
    export CUDA_NVCC_EXECUTABLE="${WORKSPACE_DIR}/nvcc"
  fi

  export PATH="${WORKSPACE_DIR}:$PATH"
fi

# If we run too many parallel jobs, we will OOM
MAX_JOBS=2 USE_DISTRIBUTED=1 python setup.py install

assert_git_not_dirty

# Upload torch binaries when the build job is finished
if [ -z "${IN_CIRCLECI}" ]; then
  7z a ${IMAGE_COMMIT_TAG}.7z ${WORKSPACE_DIR}/miniconda3/lib/python3.6/site-packages/torch*
  aws s3 cp ${IMAGE_COMMIT_TAG}.7z s3://ossci-macos-build/pytorch/${IMAGE_COMMIT_TAG}.7z --acl public-read
fi
