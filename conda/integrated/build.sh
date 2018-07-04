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
export PYTORCH_BINARY_BUILD=1
export TH_BINARY_BUILD=1
export PYTORCH_BUILD_VERSION=$PKG_VERSION
export PYTORCH_BUILD_NUMBER=$PKG_BUILDNUM

# Pytorch CUDA flags
if [[ -n $CUDA_VERSION ]]; then
  if [[ $CUDA_VERSION == 9* ]]; then
    # compile for Kepler, Kepler+Tesla, Maxwell, Pascal, Volta
    export TORCH_CUDA_ARCH_LIST="3.5;5.2+PTX;6.0;6.1;7.0"
  else
    # don't compile for Volta
    export TORCH_CUDA_ARCH_LIST="3.5;5.2+PTX;6.0;6.1"
  fi
  export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
  export NCCL_ROOT_DIR=/usr/local/cuda
  #export USE_STATIC_CUDNN=1
  #export USE_STATIC_NCCL=1
  #export ATEN_STATIC_CUDA=1
else
  export NO_CUDA=1
fi


###########################################################
# Build Caffe2
###########################################################
cmake_args=()
cmake_args+=("-DCMAKE_INSTALL_PREFIX=$PREFIX")

# Build Caffe2
mkdir -p caffe2_build && pushd caffe2_build
cmake "${cmake_args[@]}" $CAFFE2_CMAKE_ARGS ..
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
# Copies libnvrtc and libnvToolsExt to the site-packages/torch/lib/ directory
# All other CUDA libraries should be statically linked
if [[ -z $CUDA_VERSION || -z $PACKAGE_CUDA_LIBS ]]; then
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

# full_path_to_dependency <parent> <target> finds the full path to
# libtarget.so* in 'ldd libparent*.so*'
full_path_to_dependency() {
  local _DEP="$(find $SP_DIR/torch/ -name "${1}*.so*" -maxdepth 1)"
  local _TAR="lib${2}.so"
  echo $(ldd $_DEP | grep -oP '(?<= => )\S+'$_TAR'\S*')
}

# These are all the CUDA related libaries needed by Pytorch and Caffe2 that are
# not statically linked
DEPS_SOPATHS=()
DEPS_SOPATHS+=($(full_path_to_dependency '_C' 'nvToolsExt'))
DEPS_SOPATHS+=($(full_path_to_dependency '_nvrtc' 'nvrtc'))
# TODO add nvrtc-builtins too, but that doesn't show up in ldd or in patchelf

# Loop through .so, adding hashes and copying them to site-packages
patched=()
for filepath in "${DEPS_SOPATHS[@]}"
do
  filename=$(basename $filepath)
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

# Run patchelf to fix all the libaries to use the hashed names
for ((i=0;i<${#DEPS_SOPATHS[@]};++i));
do
	find $SP_DIR/torch -name '*.so*' | while read sofile; do
    origname=$(basename ${DEPS_SOPATHS[i]})
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
