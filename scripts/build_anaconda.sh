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
  portable_sed "/$1/d" $meta_yaml
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
  add_before "# ${1} section here" "$2" $meta_yaml
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
conda_args=()
caffe2_cmake_args=()
conda_channel=()
if [[ $BUILD_ENVIRONMENT == *cuda* ]]; then
  cuda_ver="$(echo $BUILD_ENVIRONMENT | grep --only-matching -P '(?<=cuda)[0-9]\.[0-9]')"
fi
if [[ $BUILD_ENVIRONMENT == *cudnn* ]]; then
  cudnn_ver="$(echo $BUILD_ENVIRONMENT | grep --only-matching -P '(?<=cudnn)[0-9](\.[0-9])?')"
fi
if [[ $BUILD_ENVIRONMENT == *full* ]]; then
  build_full=1
fi

# Support legacy way of passing in these parameters
if [[ -n $SKIP_CONDA_TESTS ]]; then
  conda_args+=("--no-test")
  conda_args+=("--no-anaconda-upload")
  upload_to_conda=''
fi
if [[ -n $UPLOAD_TO_CONDA ]]; then
  upload_to_conda=1
fi
if [[ -n $CONDA_INSTALL_LOCALLY ]]; then
  install_locally=1
fi
if [[ -n $BUILD_INTEGRATED ]]; then
  pytorch_too=1
fi

# Parameters passed in by command line. These override those set by environment
# variables
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      shift
      package_name=$1
      ;;
    --suffix)
      shift
      package_name_suffix=$1
      ;;
    --skip-tests)
      conda_args+=("--no-test")
      conda_args+=("--no-anaconda-upload")
      upload_to_conda=''
      ;;
    --upload)
      upload_to_conda=1
      ;;
    --install-locally)
      install_locally=1
      ;;
    --cuda)
      shift
      cuda_ver="$1"
      ;;
    --cudnn)
      shift
      cudnn_ver="$1"
      ;;
    --full)
      build_full=1
      ;;
    --integrated)
      pytorch_too=1
      ;;
    --pytorch-too)
      pytorch_too=1
      ;;
    --slim)
      slim=1
      ;;
    --conda)
      shift
      conda_args+=("$1")
      ;;
    *)
      caffe2_cmake_args+=("$1")
      ;;
  esac
  shift
done

# Verify that the CUDA version is supported
if [[ -n $cuda_ver ]]; then
  if [[ $cuda_ver == 9.1* ]]; then
    cuda_feature_name=cuda91
  elif [[ $cuda_ver == 9.0* ]]; then
    cuda_feature_name=cuda90
  elif [[ $cuda_ver == 8.0* ]]; then
    cuda_feature_name=cuda80
  else
    echo "Unsupported CUDA version $cuda_ver"
    exit 1
  fi
  if [[ -z $cudnn_ver ]]; then
    echo "No CuDNN version given. Caffe2 will still build against whatever"
    echo "CuDNN that it finds first, and will break if there is no CuDNN found."
  fi
  echo "Detected CUDA version $cuda_ver"
fi

###########################################################
# Set the build version
export PYTORCH_BUILD_DATE="$(date +"%Y.%m.%d")"
if [[ -n $pytorch_too ]]; then
  export PYTORCH_BUILD_VERSION="$(date +"%Y.%m.%d")"
else
  export PYTORCH_BUILD_VERSION="0.8.dev"
fi


###########################################################
# Read the gcc version to see what ABI to build for
if [[ "$(uname)" != 'Darwin' ]]; then
  gcc_ver="$(gcc --version | grep --only-matching '[0-9]\.[0-9]\.[0-9]*' | head -1)"
fi
if [[ $gcc_ver == 4* ]]; then
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
  conda_args+=(" --python 3.6")
fi


###########################################################
# Pick the correct conda-build folder
###########################################################
# And copy the meta.yaml to the correct build folder
pytorch_root="$( cd "$(dirname "$0")"/.. ; pwd -P)"
build_dir="${pytorch_root}/conda"
if [[ -n $pytorch_too ]]; then
  \cp -r "${build_dir}/caffe2/meta.yaml" "${build_dir}/integrated/meta.yaml"
  build_dir="${build_dir}/integrated"
elif [[ -n $build_full ]]; then
  build_dir="${build_dir}/caffe2/full"
else
  \cp -r "${build_dir}/caffe2/meta.yaml" "${build_dir}/caffe2/normal/meta.yaml"
  build_dir="${build_dir}/caffe2/normal"
fi
meta_yaml="${build_dir}/meta.yaml"
portable_sed "s#path:.*#path: $pytorch_root#" $meta_yaml


###########################################################
# Build the package name and build string depending on gcc and CUDA
###########################################################
build_string='py{{py}}'
if [[ -z $package_name ]]; then
  package_name='caffe2'
  if [[ -n $pytorch_too ]]; then
    package_name="pytorch-${package_name}"
  fi
fi
if [[ -n $cuda_ver ]]; then
  # CUDA 9.0 and 9.1 are not in conda, and cuDNN is not in conda, so instead of
  # pinning CUDA and cuDNN versions in the conda_build_config and then setting
  # the package name in meta.yaml based off of these values, we let Caffe2
  # take the CUDA and cuDNN versions that it finds in the build environment,
  # and manually set the package name ourself.
  package_name="${package_name}-cuda${cuda_ver}-cudnn${cudnn_ver}"
  build_string="${build_string}-cuda${cuda_ver}-cudnn${cudnn_ver}-nccl2"
else
  build_string="${build_string}-cpu"
fi
if [[ "$(uname)" != 'Darwin' && $GCC_USE_C11 -eq 0 ]]; then
  # gcc compatibility is not tracked by conda-forge, so we track it ourselves
  package_name="${package_name}-gcc${gcc_ver:0:3}"
  build_string="${build_string}-gcc${gcc_ver:0:3}"
fi
if [[ -n $build_full ]]; then
  package_name="${package_name}-full"
  build_string="${build_string}-full"
fi
portable_sed "s/name: caffe2.*\$/name: ${package_name}/" $meta_yaml
#portable_sed "s/string:.*\$/string: ${build_string}/" $meta_yaml


###########################################################
# Handle tests
###########################################################
if [[ -n $pytorch_too ]]; then
  # Removed until https://github.com/conda/conda/issues/7245 is resolved
  #if [[ -n $cuda_ver ]]; then
  #  append_to_section 'test' 'requires:'
  #  append_to_section 'test' "  - $cuda_feature_name"
  #  append_to_section 'test' '  - nccl2'
  #fi
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
caffe2_cmake_args+=("-DUSE_LEVELDB=OFF")
caffe2_cmake_args+=("-DUSE_LMDB=OFF")


# Add packages required for pytorch
if [[ -n $pytorch_too ]]; then
  add_package 'cffi'
  add_package 'mkl' '>=2018'
  add_package 'mkl-include'
  add_package 'typing'
  append_to_section 'build' '- pyyaml'
  append_to_section 'build' '- setuptools'
  #caffe2_cmake_args+=("-DBLAS=MKL")
  if [[ -n $cuda_ver ]]; then
    # Removed until https://github.com/conda/conda/issues/7245 is resolved
    #append_to_section 'features' features:
    #append_to_section 'features' "  - $cuda_feature_name" 
    append_to_section 'build' "- magma-$cuda_feature_name"
    #append_to_section 'features' '  - nccl2'
    #add_package $cuda_feature_name
    conda_channel+=('-c pytorch')

    caffe2_cmake_args+=("-DUSE_ATEN=ON")
  fi
fi

if [[ -z $slim ]]; then
  add_package 'opencv'
else
  caffe2_cmake_args+=("-DUSE_OPENCV=OFF")
fi

# Flags required for CUDA for Caffe2
if [[ -n $cuda_ver ]]; then
  caffe2_cmake_args+=("-DUSE_CUDA=ON")
  caffe2_cmake_args+=("-DUSE_NCCL=ON")

  # NCCL and GLOO don't work with static CUDA right now. Cmake changes are
  # needed
  #caffe2_cmake_args+=("-DUSE_NCCL=OFF")
  #caffe2_cmake_args+=("-DUSE_GLOO=OFF")
  #caffe2_cmake_args+=("-DCAFFE2_STATIC_LINK_CUDA=ON")

  if [[ $upload_to_conda ]]; then
    caffe2_cmake_args+=("-DCUDA_ARCH_NAME=All")
  fi
else
  # Flags required for CPU for Caffe2
  caffe2_cmake_args+=("-DUSE_CUDA=OFF")
  caffe2_cmake_args+=("-DUSE_NCCL=OFF")
  #if [[ -z $pytorch_too ]]; then
  #  #caffe2_cmake_args+=("-DBLAS=MKL")
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

  # Default conda channels use gcc 7.2, conda-forge uses gcc 4.8.5
  caffe2_cmake_args+=("-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0")
  conda_channel+=('-c conda-forge')
fi


###########################################################
# Set flags needed for uploading to Anaconda. This is only allowed if testing
# is enabled
###########################################################
if [[ $upload_to_conda ]]; then
  conda_args+=(" --user ${ANACONDA_USERNAME}")
  conda_args+=(" --token ${CAFFE2_ANACONDA_ORG_ACCESS_TOKEN}")

  # If building a redistributable, then package the CUDA libraries with it
  # TODO this doesn't work on Ubuntu right now
  #if [[ -n $cuda_ver ]]; then
  #  export PACKAGE_CUDA_LIBS=1
  #fi
fi

# Show what the final meta.yaml looks like
echo "Finalized meta.yaml is"
cat $meta_yaml


###########################################################
# Build Caffe2 with conda-build
###########################################################
CAFFE2_CMAKE_ARGS=${caffe2_cmake_args[@]} CUDA_VERSION=$cuda_ver conda build $build_dir ${conda_channel[@]} ${conda_args[@]}

# Install Caffe2 from the built package into the local conda environment
if [[ -n $install_locally ]]; then
  conda install -y ${conda_channel[@]} $package_name --use-local
fi
