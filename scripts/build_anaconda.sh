#!/bin/bash

# NOTE: All parameters to this function are forwared directly to conda-build
# and so will never be seen by the build.sh
# TODO change arguments to go to cmake by default
# TODO handle setting flags in build.sh too

set -ex

# portable_sed: A wrapper around sed that works on both mac and linux, used to
# alter conda-build files such as the meta.yaml. It always adds the inplace
# flag
#   portable_sed <full regex string> <file>
portable_sed () {
  if [ "$(uname)" == 'Darwin' ]; then
    sed -i '' "$1" "$2"
  else
    sed -i "$1" "$2"
  fi
}

# remove_package: Given a package name, removes any line that mentions that
# file from the meta.yaml
remove_package () {
  portable_sed "/$1/d" "${META_YAML}"
}

# add_package: Takes a package name and a version and finagles the
# meta.yaml to ask for that version specifically.
# NOTE: this assumes that $META_YAML has already been set
add_package () {
  remove_package $1
  # This magic string _M_STR is in the requirements sections of the meta.yaml
  # The \\"$'\n' is a properly escaped new line
  # Those 4 spaces are there to properly indent the comment
  local _M_STR='# other packages here'
  portable_sed "s/$_M_STR/- ${1} ${2}\\"$'\n'"    $_M_STR/" "${META_YAML}"
}

CAFFE2_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
CONDA_BUILD_ARGS=()
CMAKE_BUILD_ARGS=()

# Reinitialize submodules
git submodule update --init


#
# Read python and gcc version
#

# Read the gcc version to see what ABI to build for
if [[ $BUILD_ENVIRONMENT == *gcc4.8* ]]; then
  GCC_USE_C11=0
  GCC_VERSION='4.8'
fi
if [ "$(uname)" != 'Darwin' -a -z "${GCC_USE_C11}" ]; then
  GCC_VERSION="$(gcc --version | grep --only-matching '[0-9]\.[0-9]\.[0-9]*' | head -1)"
  if [[ "$GCC_VERSION" == 4* ]]; then
    GCC_USE_C11=0
  else
    GCC_USE_C11=1
  fi
fi

# Read the python version
# Specifically 3.6 because the latest Anaconda version is 3.6, and so it's site
# packages have 3.6 in the name
PYTHON_VERSION="$(python --version 2>&1 | grep --only-matching '[0-9]\.[0-9]\.[0-9]*')"
if [[ "$PYTHON_VERSION" == 3.6* ]]; then
  # This is needed or else conda tries to move packages to python3/site-packages
  # isntead of python3.6/site-packages
  CONDA_BUILD_ARGS+=(" --python 3.6")
fi


#
# Pick the correct conda-build folder
#
CAFFE2_CONDA_BUILD_DIR="${CAFFE2_ROOT}/conda/caffe2"
if [[ "${BUILD_ENVIRONMENT}" == *full* ]]; then
  CAFFE2_CONDA_BUILD_DIR="${CAFFE2_CONDA_BUILD_DIR}/cuda_full"
elif [[ "${BUILD_ENVIRONMENT}" == *cuda* ]]; then
  CAFFE2_CONDA_BUILD_DIR="${CAFFE2_CONDA_BUILD_DIR}/cuda"
else
  CAFFE2_CONDA_BUILD_DIR="${CAFFE2_CONDA_BUILD_DIR}/no_cuda"
fi
META_YAML="${CAFFE2_CONDA_BUILD_DIR}/meta.yaml"
CONDA_BUILD_CONFIG_YAML="${CAFFE2_CONDA_BUILD_DIR}/conda_build_config.yaml"


#
# Build the name of the package depending on CUDA and gcc
#
CAFFE2_PACKAGE_NAME="caffe2"
if [[ $BUILD_ENVIRONMENT == *cuda* ]]; then
  # CUDA 9.0 and 9.1 are not in conda, and cuDNN is not in conda, so instead of
  # pinning CUDA and cuDNN versions in the conda_build_config and then setting
  # the package name in meta.yaml based off of these values, we let Caffe2
  # take the CUDA and cuDNN versions that it finds in the build environment,
  # and manually set the package name ourself.
  CAFFE2_PACKAGE_NAME="${CAFFE2_PACKAGE_NAME}-cuda${CAFFE2_CUDA_VERSION}-cudnn${CAFFE2_CUDNN_VERSION}"
fi
if [[ $GCC_USE_C11 -eq 0 ]]; then
  # gcc compatibility is not tracked by conda-forge, so we track it ourselves
  CAFFE2_PACKAGE_NAME="${CAFFE2_PACKAGE_NAME}-gcc${GCC_VERSION:0:3}"
fi
if [[ $BUILD_ENVIRONMENT == *full* ]]; then
  CAFFE2_PACKAGE_NAME="${CAFFE2_PACKAGE_NAME}-full"
fi
portable_sed "s/name: caffe2.*\$/name: ${CAFFE2_PACKAGE_NAME}/" "${META_YAML}"


#
# Handle skipping tests and uploading
#
# If skipping tests, remove the test related lines from the meta.yaml and don't
# upload to Anaconda.org
if [ -n "$SKIP_CONDA_TESTS" ]; then
  portable_sed '/test:/d' "${META_YAML}"
  portable_sed '/imports:/d' "${META_YAML}"
  portable_sed '/caffe2.python.core/d' "${META_YAML}"

elif [ -n "$UPLOAD_TO_CONDA" ]; then
  # Upload to Anaconda.org if needed. This is only allowed if testing is
  # enabled
  CONDA_BUILD_ARGS+=(" --user ${ANACONDA_USERNAME}")
  CONDA_BUILD_ARGS+=(" --token ${CAFFE2_ANACONDA_ORG_ACCESS_TOKEN}")
fi


#
# Change flags based on target gcc ABI
#
if [[ "$(uname)" != 'Darwin' ]]; then
  if [ "$GCC_USE_C11" -eq 0 ]; then
    CMAKE_BUILD_ARGS+=("-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0")
    # Default conda channels use gcc 7.2 (for recent packages), conda-forge uses
    # gcc 4.8.5
    # TODO don't do this if user also specified a channel
    CAFFE2_CONDA_CHANNEL='-c conda-forge'

    # opencv 3.3.1 in conda-forge doesn't have imgcodecs
    add_package 'opencv' '==3.1.0'
    if [[ "$PYTHON_VERSION" == 3.* ]]; then
      # opencv 3.1.0 for python 3 requires numpy 1.12
      add_package 'numpy' '>1.11'
    fi
    CONDA_BUILD_ARGS+=(" -c conda-forge")

  else
    # gflags 2.2.1 is built against the new ABI but gflags 2.2.0 is not
    add_package 'gflags' '==2.2.1'

    # opencv 3.3.1 requires protobuf 3.2.0 explicitly, so we use opencv 3.1.0
    # since protobuf 3.2.0 is not in conda
    add_package 'opencv' '==3.1.0'
    if [[ "$PYTHON_VERSION" == 3.* ]]; then
      # opencv 3.1.0 for python 3 requires numpy 1.12
      add_package 'numpy' '>1.11'
    fi

    # These calls won't work since
    #  - these package requirements can't be put in meta.yaml (no support yet)
    #  - if they're put here then they won't be installed at test or install
    #      time
    # glog 0.3.5=0 is built against old ABI, but 0.3.5=hf484d3e_1 is not
    #remove_package 'glog'
    #conda install -y 'glog=0.3.5=hf484d3e_1'

    # leveldb=1.20 is built against old ABI, but 1.20=hf484d3e_1 is built
    # against the new one
    #remove_package 'leveldb'
    #conda install -y 'leveldb=1.20=hf484d3e_1'
  fi
else
  # On macOS opencv 3.3.1 (there's only 3.3.1 and 2.4.8) requires protobuf
  # 3.4
  portable_sed "s/3.5.1/3.4.1/" "${CONDA_BUILD_CONFIG_YAML}"
fi


#

# Build Caffe2 with conda-build
#
# If --user and --token are set, then this will also upload the built package
# to Anaconda.org, provided there were no failures and all the tests passed
CONDA_CMAKE_BUILD_ARGS="$CMAKE_BUILD_ARGS" conda build "${CAFFE2_CONDA_BUILD_DIR}" $CAFFE2_CONDA_CHANNEL ${CONDA_BUILD_ARGS[@]} "$@"

# Install Caffe2 from the built package into the local conda environment
if [ -n "$CONDA_INSTALL_LOCALLY" ]; then
  conda install -y $CAFFE2_CONDA_CHANNEL "${CAFFE2_PACKAGE_NAME}" --use-local
fi
