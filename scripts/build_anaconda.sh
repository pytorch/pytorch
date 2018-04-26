#!/bin/bash

# This script creates and (possibly) uploads a Caffe2 Anaconda package, and
# then (optionally) installs the package locally into the current activated
# conda environment. This script handles flags needed by CUDA and gcc versions,
# and has good default behavior if called with no arguments.
#
# Usage:
#  ./build_anaconda.sh [--cuda X.Y] [--cudnn Z]
#                      [--full] [--integrated]
#                      [--conda <flag forwared to conda-build>]...
#                      [<flags forwarded to cmake>]...
#
# Parameters can also be passed through the BUILD_ENVIRONMENT environment
# variable, e.g. 
#  BUILD_ENVIRONMENT=conda2-cuda8.0-cudnn7-gcc4.8 ./scripts/build_anaconda.sh
# - Parameters parsed from the BUILD_ENVIRONMENT will be overridden by command
#   line parameters.
# - The conda version and gcc version given in BUILD_ENVIRONMENT are ignored.
#   These versions are determined by calling the binaries with --version
#
# The special values CAFFE2_ANACONDA_ORG_ACCESS_TOKEN, and
# ANACONDA_USERNAME can only be passed in as environment variables.
#
# This script works by
# 1. Choosing the correct conda-build folder to use
# 2. Building the package name and build-string
# 3. Determining which flags and packages are required
# 4. Calling into conda-build
# 5. (optional) installing the built package locally

set -ex

#
# Functions used in this script
#

# portable_sed: A wrapper around sed that works on both mac and linux, used to
# alter conda-build files such as the meta.yaml. It always adds the inplace
# flag
#   portable_sed <full regex string> <file>
if [ "$(uname)" == 'Darwin' ]; then
  portable_sed () {
    sed -E -i '' "$1" "$2"
  }
else
  portable_sed () {
    sed --regexp-extended -i "$1" "$2"
  }
fi

# remove_lines_with: Given a string, removes any line that contains it
remove_lines_with () {
  portable_sed "/$1/d" $META_YAML
}

# add_before <some marker> <some insertion> <in this file>
# essentially replaces
#
#    <some marker>
#
# with
#
#    <some insertion>
#    <some marker>
#
# ( *)     captured spaces before match == the indentation in the meta.yaml
# ${1}     the marker to insert before
# '\1'     captured whitespace == correct indentation
# ${2}     the string to insert
# \\"$'\n' escaped newline
# '\1'      captured whitespace == correct indentation
# ${1}     put the marker back
add_before() {
  portable_sed 's@( *)'"${1}@"'\1'"${2}\\"$'\n''\1'"${1}@" $3
}
append_to_section () {
  add_before "# ${1} section here" "$2" $META_YAML
}
# add_package <package_name> <optional package version specifier>
# Takes a package name and version and finagles the meta.yaml to specify that
add_package () {
  append_to_section 'build' "- $1 $2"
  append_to_section 'run' "- $1 $2"
}


###########################################################
# Parse options from both command line and from BUILD_ENVIRONMENT
###########################################################
CONDA_BUILD_ARGS=()
CAFFE2_CMAKE_ARGS=()
CONDA_CHANNEL=()
if [[ $BUILD_ENVIRONMENT == *cuda* ]]; then
  CUDA_VERSION="$(echo $BUILD_ENVIRONMENT | grep --only-matching -P '(?<=cuda)[0-9]\.[0-9]')"
fi
if [[ $BUILD_ENVIRONMENT == *cudnn* ]]; then
  CUDNN_VERSION="$(echo $BUILD_ENVIRONMENT | grep --only-matching -P '(?<=cudnn)[0-9](\.[0-9])?')"
fi
if [[ $BUILD_ENVIRONMENT == *full* ]]; then
  BUILD_FULL=1
fi

# Support legacy way of passing in these parameters
if [[ -n $SKIP_CONDA_TESTS ]]; then
  skip_tests=1
fi
if [[ -n $UPLOAD_TO_CONDA ]]; then
  upload_to_conda=1
fi
if [[ -n $CONDA_INSTALL_LOCALLY ]]; then
  install_locally=1
fi

# Parameters passed in by command line. These override those set by environment
# variables
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      shift
      PACKAGE_NAME=$1
      ;;
    --suffix)
      shift
      package_name_suffix=$1
      ;;
    --skip-tests)
      skip_tests=1
      ;;
    --upload)
      upload_to_conda=1
      ;;
    --install-locally)
      install_locally=1
      ;;
    --cuda)
      shift
      CUDA_VERSION="$1"
      ;;
    --cudnn)
      shift
      CUDNN_VERSION="$1"
      ;;
    --full)
      BUILD_FULL=1
      ;;
    --integrated)
      BUILD_INTEGRATED=1
      ;;
    --conda)
      shift
      CONDA_BUILD_ARGS+=("$1")
      ;;
    *)
      CAFFE2_CMAKE_ARGS+=("$1")
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
  echo "Detected CUDA_VERSION $CUDA_VERSION"
fi

# Only allow uploading to Anaconda if the tests were run
if [[ -n $skip_tests && -n $upload_to_conda ]]; then
  echo "Uploading to Anaconda only allowed if tests are run. Upload turned off"
  upload_to_conda=''
fi

###########################################################
# Set the build version
if [[ -n $BUILD_INTEGRATED ]]; then
  export PYTORCH_BUILD_VERSION="$(date +"%Y.%m.%d")"
else
  export PYTORCH_BUILD_VERSION="0.8.dev.$(date +"%Y.%m.%d")"
fi


###########################################################
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
PYTHON_VERSION="$(python --version 2>&1 | grep --only-matching '[0-9]\.[0-9]\.[0-9]*')"
if [[ "$PYTHON_VERSION" == 3.6* ]]; then
  # This is needed or else conda tries to move packages to python3/site-packages
  # instead of python3.6/site-packages. Specifically 3.6 because that's what
  # the latest Anaconda version is
  CONDA_BUILD_ARGS+=(" --python 3.6")
fi


###########################################################
# Pick the correct conda-build folder
###########################################################
PYTORCH_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
CONDA_BUILD_DIR="${PYTORCH_ROOT}/conda"
if [[ -n $BUILD_INTEGRATED ]]; then
  # This cp is so only one meta.yaml is used
  \cp -r "$CONDA_BUILD_DIR/caffe2/normal/meta.yaml" "$CONDA_BUILD_DIR/integrated/meta.yaml"
  CONDA_BUILD_DIR="${CONDA_BUILD_DIR}/integrated"
elif [[ -n $BUILD_FULL ]]; then
  CONDA_BUILD_DIR="${CONDA_BUILD_DIR}/caffe2/full"
else
  CONDA_BUILD_DIR="${CONDA_BUILD_DIR}/caffe2/normal"
fi
META_YAML="${CONDA_BUILD_DIR}/meta.yaml"
portable_sed "s#path:.*#path: $PYTORCH_ROOT#" $META_YAML


###########################################################
# Build the package name and build string depending on gcc and CUDA
###########################################################
BUILD_STRING='py{{py}}'
if [[ -z $PACKAGE_NAME ]]; then
  PACKAGE_NAME='caffe2'
  if [[ -n $BUILD_INTEGRATED ]]; then
    PACKAGE_NAME="pytorch-${PACKAGE_NAME}"
  fi
fi
if [[ -n $CUDA_VERSION ]]; then
  # CUDA 9.0 and 9.1 are not in conda, and cuDNN is not in conda, so instead of
  # pinning CUDA and cuDNN versions in the conda_build_config and then setting
  # the package name in meta.yaml based off of these values, we let Caffe2
  # take the CUDA and cuDNN versions that it finds in the build environment,
  # and manually set the package name ourself.
  BUILD_STRING="${BUILD_STRING}_cuda${CUDA_VERSION}_cudnn${CUDNN_VERSION}_nccl2"
else
  BUILD_STRING="${BUILD_STRING}_cpu"
fi
if [[ "$(uname)" != 'Darwin' && -z $BUILD_INTEGRATED && $GCC_USE_C11 -eq 0 ]]; then
  # gcc compatibility is not tracked by conda-forge, so we track it ourselves
  BUILD_STRING="${BUILD_STRING}_gcc${GCC_VERSION:0:3}"
fi
if [[ -n $BUILD_FULL ]]; then
  BUILD_STRING="${BUILD_STRING}_full"
fi
portable_sed "s/name: caffe2.*\$/name: ${PACKAGE_NAME}/" $META_YAML
portable_sed "s/string:.*\$/string: ${BUILD_STRING}/" $META_YAML


###########################################################
# Handle tests
###########################################################
if [[ -n $skip_tests ]]; then
  remove_lines_with 'test:'
  remove_lines_with 'imports:'
  remove_lines_with 'caffe2.python.core'
elif [[ -n $BUILD_INTEGRATED ]]; then
  if [[ -n $CUDA_VERSION ]]; then
    append_to_section 'test' 'requires:'
    append_to_section 'test' "  - $CUDA_FEATURE_NAME"
    append_to_section 'test' '  - nccl2'
  fi
  append_to_section 'test' 'source_files:'
  append_to_section 'test' '  - test'
  append_to_section 'test' 'commands:'
  append_to_section 'test' '  - OMP_NUM_THREADS=4 ./test/run_test.sh || true'
fi


###########################################################
# Set flags and package requirements
###########################################################
# Add packages required for all Caffe2 builds
add_package 'glog'
add_package 'gflags'
add_package 'opencv'

# Add packages required for pytorch
if [[ -n $BUILD_INTEGRATED ]]; then
  remove_lines_with 'numpy'
  add_package 'cffi'
  add_package 'mkl' '>=2018'
  add_package 'mkl-include'
  add_package 'numpy' '>=1.11'
  add_package 'typing'
  append_to_section 'build' '- pyyaml'
  append_to_section 'build' '- setuptools'
  CAFFE2_CMAKE_ARGS+=("-DBLAS=MKL")
  if [[ -n $CUDA_VERSION ]]; then
    append_to_section 'features' features:
    append_to_section 'features' "  - $CUDA_FEATURE_NAME" 
    append_to_section 'features' '  - nccl2'
    add_package $CUDA_FEATURE_NAME
    CONDA_CHANNEL+=('-c pytorch')
  fi
else
  add_package 'leveldb'
fi

# Flags required for CUDA for Caffe2
if [[ -n $CUDA_VERSION ]]; then
  CAFFE2_CMAKE_ARGS+=("-DUSE_CUDA=ON")
  CAFFE2_CMAKE_ARGS+=("-DUSE_NCCL=ON")

  # NCCL and GLOO don't work with static CUDA right now. Cmake changes are
  # needed
  #CAFFE2_CMAKE_ARGS+=("-DUSE_NCCL=OFF")
  #CAFFE2_CMAKE_ARGS+=("-DUSE_GLOO=OFF")
  #CAFFE2_CMAKE_ARGS+=("-DCAFFE2_STATIC_LINK_CUDA=ON")
else
  # Flags required for CPU for Caffe2
  CAFFE2_CMAKE_ARGS+=("-DUSE_CUDA=OFF")
  CAFFE2_CMAKE_ARGS+=("-DUSE_NCCL=OFF")
  #if [[ -z $BUILD_INTEGRATED ]]; then
  #  #CAFFE2_CMAKE_ARGS+=("-DBLAS=MKL")
  #  #add_package 'mkl'
  #  #add_package 'mkl-include'
  #fi
fi

# Change flags based on target gcc ABI
if [[ "$(uname)" != 'Darwin' && "$GCC_USE_C11" -eq 0 ]]; then
  # opencv 3.3.1 in conda-forge doesn't have imgcodecs, and opencv 3.1.0
  # requires numpy 1.12
  remove_lines_with 'opencv'
  add_package 'opencv' '==3.1.0'
  if [[ "$PYTHON_VERSION" == 3.* ]]; then
    remove_lines_with 'numpy'
    add_package 'numpy' '>1.11'
  fi
  # Default conda channels use gcc 7.2, conda-forge uses gcc 4.8.5
  CAFFE2_CMAKE_ARGS+=("-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0")
  CONDA_CHANNEL+=('-c conda-forge')
fi


###########################################################
# Set flags needed for uploading to Anaconda. This is only allowed if testing
# is enabled
###########################################################
if [[ $upload_to_conda ]]; then
  CONDA_BUILD_ARGS+=(" --user ${ANACONDA_USERNAME}")
  CONDA_BUILD_ARGS+=(" --token ${CAFFE2_ANACONDA_ORG_ACCESS_TOKEN}")

  # If building a redistributable, then package the CUDA libraries with it
  # TODO this doesn't work on Ubuntu right now
  #if [[ -n $CUDA_VERSION ]]; then
  #  export PACKAGE_CUDA_LIBS=1
  #fi
fi

# Show what the final meta.yaml looks like
echo "Finalized meta.yaml is"
cat $META_YAML


###########################################################
# Build Caffe2 with conda-build
###########################################################
CONDA_CAFFE2_CMAKE_ARGS=${CAFFE2_CMAKE_ARGS[@]} conda build $CONDA_BUILD_DIR ${CONDA_CHANNEL[@]} ${CONDA_BUILD_ARGS[@]} "$@"

# Install Caffe2 from the built package into the local conda environment
if [[ -n $install_locally ]]; then
  conda install -y ${CONDA_CHANNEL[@]} $PACKAGE_NAME --use-local
fi
