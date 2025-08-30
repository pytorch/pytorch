#!/bin/bash

# This script should be sourced to set environment variables specific to
# building using the ROCm Python wheels.
# See https://github.com/ROCm/TheRock/blob/main/docs/packaging/python_packaging.md
# and https://github.com/ROCm/TheRock/blob/main/external-builds/pytorch/build_prod_wheels.py

if [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
  export USE_ROCM=ON
  # TODO: add rocm[libraries]==ROCM_SDK_VERSION to PYTORCH_EXTRA_INSTALL_REQUIREMENTS?
  #       (perhaps with ~= or some other looser pin)
  # TODO: add 'rocmsdk' to PYTORCH_BUILD_VERSION?

  # Look up rocm_sdk paths.
  # TODO: fail/skip if these do not work
  ROCM_ROOT_PATH=$(python -m rocm_sdk path --root)
  ROCM_BIN_PATH=$(python -m rocm_sdk path --bin)
  ROCM_CMAKE_PATH=$(python -m rocm_sdk path --cmake)

  # Set common build environment variables using those rocm_sdk paths.
  export CMAKE_PREFIX_PATH=${ROCM_CMAKE_PATH}
  export ROCM_HOME=${ROCM_ROOT_PATH}
  export ROCM_PATH=${ROCM_ROOT_PATH}
  echo "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}"
  echo "ROCM_HOME: ${ROCM_HOME}"
  echo "ROCM_PATH: ${ROCM_PATH}"

  # TODO: prepend ROCM_BIN_PATH to PATH as needed (venv already does this?)
  # system_path = str(bin_dir) + os.path.pathsep + os.environ.get("PATH", "")

  if [[ "$BUILD_ENVIRONMENT" == *windows-* ]]; then
    # Set feature support environment variables (overriding defaults).
    export DISTUTILS_USE_SDK=1
    # TODO: enable tests
    export BUILD_TEST=0
    # TODO: enable attention (with aotriton)
    export USE_FLASH_ATTENTION=0
    export USE_MEM_EFF_ATTENTION=0
    # TODO: enable Kineto somehow
    export USE_KINETO=0
    # TODO: enable GLOO somehow
    export USE_GLOO=0

    # Use the LLVM toolchain from rocm_sdk (clang-cl, etc.).
    LLVM_DIR_WIN="${ROCM_ROOT_PATH}\\lib\\llvm\\bin"
    LLVM_DIR=$(cygpath --unix "${LLVM_DIR_WIN}")
    export HIP_CLANG_PATH=${LLVM_DIR}
    export CC="${LLVM_DIR_WIN}\\clang-cl.exe"
    export CXX="${LLVM_DIR_WIN}\\clang-cl.exe"
    HIP_DEVICE_LIB_PATH_WIN="${ROCM_ROOT_PATH}\\lib\\llvm\\amdgcn\\bitcode"
    export HIP_DEVICE_LIB_PATH=${HIP_DEVICE_LIB_PATH_WIN}
    echo "HIP_CLANG_PATH: ${HIP_CLANG_PATH}"
    echo "CC: ${CC}"
    echo "CXX: ${CXX}"
    echo "HIP_DEVICE_LIB_PATH: ${HIP_DEVICE_LIB_PATH}"
  fi

  # TODO: Linux-specific settings

  if [[ -n "$CI" && -z "$PYTORCH_ROCM_ARCH" ]]; then
      # Set ROCM_ARCH to gfx1100 for CI builds, if user doesn't override.
      echo "Limiting PYTORCH_ROCM_ARCH to gfx1100 for CI builds"
      export PYTORCH_ROCM_ARCH="gfx1100"
  fi
fi
