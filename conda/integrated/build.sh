#!/usr/bin/env bash

# conda-build script used for integrated pytorch-caffe2 packages
# NOTE: The meta.yaml in this directory should not be changed; it will be
# overwritten by the meta.yaml in conda/caffe2/normal/ when
# scripts/build_anaconda.sh is run.

# Install script for Anaconda environments on macOS and linux.
# This script is not supposed to be called directly, but should be called by
# scripts/build_anaconda.sh, which handles setting lots of needed flags
# depending on the current system and user flags.
#
# If you're debugging this, it may be useful to use the env that conda build is
# using:
# $ cd <anaconda_root>/conda-bld/caffe2_<timestamp>
# $ source activate _h_env_... # some long path with lots of placeholders
#
# Also, failed builds will accumulate those caffe2_<timestamp> directories. You
# can remove them after a succesfull build with
# $ conda build purge

set -ex

# Pytorch environment variables needed during the build
export CMAKE_LIBRARY_PATH=$PREFIX/lib:$PREFIX/include:$CMAKE_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$PREFIX
# compile for Kepler, Kepler+Tesla, Maxwell, Pascal, Volta
export TORCH_CUDA_ARCH_LIST="3.5;5.2+PTX;6.0;6.1;7.0"
export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
export PYTORCH_BINARY_BUILD=1
export TH_BINARY_BUILD=1
export PYTORCH_BUILD_VERSION=$PKG_VERSION
export PYTORCH_BUILD_NUMBER=$PKG_BUILDNUM
export NCCL_ROOT_DIR=/usr/local/cuda

# Validate some configurations, try to fail as early as possible
if [[ -n $PACKAGE_CUDA_DIR ]]; then
  if [[ -z $PACKAGE_CUDA_DIR || -z $CUDA_VERSION || -z $CUDNN_VERSION ]]; then
    echo "Packaging CUDA libs along with the Pytorch binaries is only allowed"
    echo "if PACKAGE_CUDA_DIR, CUDA_VERSION, and CUDNN_VERSION are all set."
    echo "This should only be used when building binaries for distribution"
    exit 1
  fi
fi


###########################################################
# Build Caffe2
###########################################################
PYTHON_ARGS="$(python ./scripts/get_python_cmake_flags.py)"
CMAKE_ARGS=()
CMAKE_ARGS+=("-DCMAKE_INSTALL_PREFIX=$PREFIX")

# Build Caffe2
mkdir -p caffe2_build && pushd caffe2_build
cmake "${CMAKE_ARGS[@]}" "$PYTHON_ARGS" $CONDA_CAFFE2_CMAKE_ARGS ..
if [ "$(uname)" == 'Darwin' ]; then
  make "-j$(sysctl -n hw.ncpu)"
else
  make "-j$(nproc)"
fi
make install/fast
popd




###########################################################
# Build Pytorch
###########################################################
if [[ "$OSTYPE" == "darwin"* ]]; then
  MACOSX_DEPLOYMENT_TARGET=10.9 python setup.py install
  exit 0
fi
python setup.py install



#########################################################################
# Copy over CUDA .so files from system locations to the conda build dir #
#########################################################################
if [[ -z $PACKAGE_CUDA_DIR ]]; then
  exit 0
fi

# Function to rename .so files with their hashes appended to them
fname_with_sha256() {
  HASH=$(sha256sum $1 | cut -c1-8)
  DIRNAME=$(dirname $1)
  BASENAME=$(basename $1)
  if [[ $BASENAME == "libnvrtc-builtins.so" ]]; then
	  echo $1
  else
	  INITNAME=$(echo $BASENAME | cut -f1 -d".")
	  ENDNAME=$(echo $BASENAME | cut -f 2- -d".")
	  echo "$DIRNAME/$INITNAME-$HASH.$ENDNAME"
  fi
}

# This is the install location on the Pytorch docker images used to build the
# pytorch binaries
PACKAGE_CUDA_DIR="/usr/local/cuda/lib64/"

# These are all the CUDA related libaries needed by Pytorch and Caffe2
# for some reason if we use exact version numbers for CUDA9 .so files 
# (like .so.9.0.176), we see segfaults during dlopen somewhere
# deep inside these libs.
# hence for CUDA9, use e.g. '.9.0', and don't use hashed names
DEPS_SONAME=(
  "libcudart.so.${CUDA_VERSION:0:3}"
  "libnvToolsExt.so.1"
  "libcublas.so.${CUDA_VERSION:0:3}"
  "libcurand.so.${CUDA_VERSION:0:3}"
  "libcusparse.so.${CUDA_VERSION:0:3}"
  "libnvrtc.so.${CUDA_VERSION:0:3}"
  "libnvrtc-builtins.so"
  "libcudnn.so.${CUDNN_VERSION:0:1}"
  "libnccl.so.2"
)

# Find which CUDA libraries the Pytorch binaries built against
# This should probably be done by writing these to a file in cmake when the
# files are found, and then reading that file here.
# This is needed to handle libcuda and libcudnn on CI machines, and will also
# allow this script to work for any user as well

# Loop through .so, renaming and moving all of them
patched=()
for filename in "${DEPS_SONAME[@]}"
do
	filepath="$PACKAGE_CUDA_DIR/$filename"
	destpath=$SP_DIR/torch/lib/$filename
	if [[ "$filepath" != "$destpath" ]]; then
    echo "Copying $filepath to $destpath"
	  cp $filepath $destpath
	fi

	patchedpath=$(fname_with_sha256 $destpath)
	patchedname=$(basename $patchedpath)
	if [[ "$destpath" != "$patchedpath" ]]; then
    echo "Moving $destpath to $patchedpath"
	  mv $destpath $patchedpath
	fi

	patched+=("$patchedname")
	echo "Copied $filepath to $patchedpath"
done

# run patchelf to fix the so names to the hashed names
for ((i=0;i<${#DEPS_SONAME[@]};++i));
do
	find $SP_DIR/torch -name '*.so*' | while read sofile; do
	  origname=${DEPS_SONAME[i]}
	  patchedname=${patched[i]}
	  if [[ "$origname" != "$patchedname" ]]; then
	    set +e
	    patchelf --print-needed $sofile | grep $origname 2>&1 >/dev/null
	    ERRCODE=$?
	    set -e
	    if [ "$ERRCODE" -eq "0" ]; then
	        echo "patching $sofile entry $origname to $patchedname"
	        patchelf --replace-needed $origname $patchedname $sofile
	    fi
	  fi
	done
done

# set RPATH of _C.so and similar to $ORIGIN, $ORIGIN/lib and conda/lib
find $SP_DIR/torch -name "*.so*" -maxdepth 1 -type f | while read sofile; do
	echo "Setting rpath of $sofile to " '$ORIGIN:$ORIGIN/lib:$ORIGIN/../../..'
	patchelf --set-rpath '$ORIGIN:$ORIGIN/lib:$ORIGIN/../../..' $sofile
	patchelf --print-rpath $sofile
done
    
# set RPATH of lib/ files to $ORIGIN and conda/lib
find $SP_DIR/torch/lib -name "*.so*" -maxdepth 1 -type f | while read sofile; do
	echo "Setting rpath of $sofile to " '$ORIGIN:$ORIGIN/lib:$ORIGIN/../../../..'
	patchelf --set-rpath '$ORIGIN:$ORIGIN/../../../..' $sofile
	patchelf --print-rpath $sofile
done
    
