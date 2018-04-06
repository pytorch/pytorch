#!/bin/bash

# This script creates and (possibly) uploads a Caffe2 Anaconda package, and
# then (optionally) installs the package locally into the current activated
# conda environment. This script handles flags needed by CUDA and gcc versions,
# and has good default behavior if called with no arguments.
#
# Usage:
#  ./build_anaconda.sh [--cuda X.Y] [--cudnn Z] [--conda <flag forwared to conda-build>]... [<flags forwarded to cmake>]...
#
# Parameters can also be passed through the BUILD_ENVIRONMENT environment
# variable, e.g. 
#  BUILD_ENVIRONMENT=conda2-cuda8.0-cudnn7-gcc4.8 ./scripts/build_anaconda.sh
# - Parameters parsed from the BUILD_ENVIRONMENT will be overridden by command
#   line parameters.
# - The conda version and gcc version given in BUILD_ENVIRONMENT are ignored.
#   These versions are determined by calling the binaries with --version
#
# The special flags SKIP_CONDA_TESTS, CAFFE2_ANACONDA_ORG_ACCESS_TOKEN, and
# ANACONDA_USERNAME can only be passed in as environment variables.

set -ex

#
# Functions used in this script
#

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

# remove_package: Given a string, removes any line that mentions that line from
# the meta.yaml
remove_package () {
  portable_sed "/$1/d" "${META_YAML}"
}

# add_package: Takes a package name and a version and finagles the
# meta.yaml to ask for that version specifically.
# NOTE: this assumes that $META_YAML has already been set
# The \\"$'\n' is a properly escaped new line
# Those 4 spaces are there to properly indent the comment
add_package () {
  remove_package $1
  # This magic string _M_STR is in the requirements sections of the meta.yaml
  local _M_STR='# other packages here'
  portable_sed "s/$_M_STR/- ${1} ${2}\\"$'\n'"    $_M_STR/" "${META_YAML}"
}

# add_feature: Adds a given feature tag to the build section. Takes care to
# only add the feature section once. Assumes that no feature section exists yet
add_feature() {
  local _M_STR='# features go here'
  if [[ -z $ADDED_A_FEATURE ]]; then
    portable_sed "s/$_M_STR/features:\\"$'\n'"      $_M_STR/" "${META_YAML}"
    ADDED_A_FEATURE=YES
  fi
  portable_sed "s/$_M_STR/- ${1}\\"$'\n'"      $_M_STR/" "${META_YAML}"
}


#
# Parse options from both command line and from BUILD_ENVIRONMENT
#
CONDA_BUILD_ARGS=()
CMAKE_BUILD_ARGS=()
if [[ $BUILD_ENVIRONMENT == *cuda* ]]; then
  CUDA_VERSION="$($BUILD_ENVIRONMENT | grep --only-matching '(?<=cuda)[0-9]\.[0-9]')"
fi
if [[ $BUILD_ENVIRONMENT == *cudnn* ]]; then
  CUDNN_VERSION="$($BUILD_ENVIRONMENT | grep --only-matching '(?<=cudnn)[0-9](\.[0-9])?')"
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)
      shift
      CUDA_VERSION="$1"
      ;;
    --cudnn)
      shift
      CUDNN_VERSION="$1"
      ;;
    --conda)
      shift
      CONDA_BUILD_ARGS+=("$1")
      ;;
    *)
      CMAKE_BUILD_ARGS+=("$1")
      ;;
  esac
  shift
done

# Verify that the CUDA version is supported
if [[ -n $CUDA_VERSION ]]; then
  if [[ $CUDA_VERSION == 9.1* ]]; then
    CUDA_FEATURE_NAME=cuda91
  elif [[ $CUDA_VERSION == 9.0* ]]; then
    CUDA_FEATURE_NAME=cuda90
  elif [[ $CUDA_VERSION == 8.0* ]]; then
    CUDA_FEATURE_NAME=cuda80
  else
    echo "Unsupported CUDA version $CUDA_VERSION"
    echo "Changes have already been made to the meta.yaml, you may have to revert them"
    exit 1
  fi
  if [[ -z $CUDNN_VERSION ]]; then
    echo "No CuDNN version given. Caffe2 will still build against whatever"
    echo "CuDNN that it finds first, and will break if there is no CuDNN found."
  fi
  echo "Detected CUDA_VERSION of $CUDA_VERSION"
fi


#
# Read python and gcc version
#
# Read the gcc version to see what ABI to build for
if [[ "$(uname)" != 'Darwin' ]]; then
  GCC_VERSION="$(gcc --version | grep --only-matching '[0-9]\.[0-9]\.[0-9]*' | head -1)"
fi
if [[ "$GCC_VERSION" == 4* ]]; then
  GCC_USE_C11=0
else
  GCC_USE_C11=1
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
PYTORCH_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
CAFFE2_CONDA_BUILD_DIR="${PYTORCH_ROOT}/conda/caffe2"
if [[ "${BUILD_ENVIRONMENT}" == *full* ]]; then
  CAFFE2_CONDA_BUILD_DIR="${CAFFE2_CONDA_BUILD_DIR}/full"
else
  CAFFE2_CONDA_BUILD_DIR="${CAFFE2_CONDA_BUILD_DIR}/normal"
fi
META_YAML="${CAFFE2_CONDA_BUILD_DIR}/meta.yaml"


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
if [[ "$(uname)" != 'Darwin' ]]; then
  if [[ $GCC_USE_C11 -eq 0 ]]; then
    # gcc compatibility is not tracked by conda-forge, so we track it ourselves
    CAFFE2_PACKAGE_NAME="${CAFFE2_PACKAGE_NAME}-gcc${GCC_VERSION:0:3}"
  fi
fi
if [[ $BUILD_ENVIRONMENT == *full* ]]; then
  CAFFE2_PACKAGE_NAME="${CAFFE2_PACKAGE_NAME}-full"
fi
portable_sed "s/name: caffe2.*\$/name: ${CAFFE2_PACKAGE_NAME}/" "${META_YAML}"


#
# Handle skipping tests and uploading built packages to Anaconda.org
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
# Set flags and package requirements
#
# Add normal packages for all builds
add_package glog
add_package gflags
add_package leveldb
add_package lmdb
add_package opencv
#
# Change flags based on target gcc ABI
#
if [[ "$(uname)" != 'Darwin' ]]; then
  if [ "$GCC_USE_C11" -eq 0 ]; then
    # opencv 3.3.1 in conda-forge doesn't have imgcodecs, and opencv 3.1.0
    # requires numpy 1.12
    add_package 'opencv' '==3.1.0'
    if [[ "$PYTHON_VERSION" == 3.* ]]; then
      add_package 'numpy' '>1.11'
    fi
    # Default conda channels use gcc 7.2 (for recent packages), conda-forge uses
    # gcc 4.8.5
    CMAKE_BUILD_ARGS+=("-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0")
    CAFFE2_CONDA_CHANNEL='-c conda-forge'
  fi
fi
#
# Handle CUDA related flags
#
if [[ -n $CUDA_VERSION ]]; then
  # Only CUDA versions 9.1, 9.0, and 8.0 are supported
  CMAKE_BUILD_ARGS+=("-DUSE_CUDA=ON")
  CMAKE_BUILD_ARGS+=("-DUSE_NCCL=ON")
  #add_feature $CUDA_FEATURE_NAME
  #add_package $CUDA_FEATURE_NAME
  #add_feature nccl2
else
  CMAKE_BUILD_ARGS+=("-DUSE_CUDA=OFF")
  CMAKE_BUILD_ARGS+=("-DUSE_NCCL=OFF")
  CMAKE_BUILD_ARGS+=("-DBLAS=MKL")
  add_package 'mkl'
  add_package 'mkl-include'
fi


#
# Build Caffe2 with conda-build
#
# If --user and --token are set, then this will also upload the built package
# to Anaconda.org, provided there were no failures and all the tests passed
CONDA_CMAKE_BUILD_ARGS="${CMAKE_BUILD_ARGS[@]}" conda build "${CAFFE2_CONDA_BUILD_DIR}" $CAFFE2_CONDA_CHANNEL ${CONDA_BUILD_ARGS[@]} "$@"

# Install Caffe2 from the built package into the local conda environment
if [ -n "$CONDA_INSTALL_LOCALLY" ]; then
  conda install -y $CAFFE2_CONDA_CHANNEL "${CAFFE2_PACKAGE_NAME}" --use-local
fi
