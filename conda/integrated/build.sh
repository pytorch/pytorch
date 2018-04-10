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

# Pytorch environment variables
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


###########################################################
# Build Caffe2
###########################################################
PYTHON_ARGS="$(python ./scripts/get_python_cmake_flags.py)"

# Install under specified prefix
CMAKE_ARGS=()
CMAKE_ARGS+=("-DCMAKE_INSTALL_PREFIX=$PREFIX")
CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$PREFIX")

# Build Caffe2
mkdir -p caffe2_build
cd caffe2_build
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

# Install
python setup.py install

###########################################################
# Copy over CUDA .so files from system locations to the conda build dir
###########################################################
if [[ -z $PACKAGE_CUDA_LIBS ]]; then
  exit 0
fi

# Rename .so files with their hashes included
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

# for some reason if we use exact version numbers for CUDA9 .so files 
# (like .so.9.0.176), we see segfaults during dlopen somewhere
# deep inside these libs.
# hence for CUDA9, use .9.0, and dont use hashed names
DEPS_LIST=(
    "/usr/local/cuda/lib64/libcudart.so.9.0"
    "/usr/local/cuda/lib64/libnvToolsExt.so.1"
    "/usr/local/cuda/lib64/libcublas.so.9.0"
    "/usr/local/cuda/lib64/libcurand.so.9.0"
    "/usr/local/cuda/lib64/libcusparse.so.9.0"
    "/usr/local/cuda/lib64/libnvrtc.so.9.0"
    "/usr/local/cuda/lib64/libnvrtc-builtins.so"
    "/usr/local/cuda/lib64/libcudnn.so.7"
    "/usr/local/cuda/lib64/libnccl.so.2"
)

DEPS_SONAME=(
    "libcudart.so.9.0"
    "libnvToolsExt.so.1"
    "libcublas.so.9.0"
    "libcurand.so.9.0"
    "libcusparse.so.9.0"
    "libnvrtc.so.9.0"
    "libnvrtc-builtins.so"
    "libcudnn.so.7"
    "libnccl.so.2"
)

# Loop through .so, renaming and moving all of them
patched=()
for filepath in "${DEPS_LIST[@]}"
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

# run patchelf to fix the so names to the hashed names
for ((i=0;i<${#DEPS_LIST[@]};++i));
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
    
