#!/bin/bash
set -eux -o pipefail

echo ""
echo "PWD: ${PWD}"
WORKSPACE=/Users/distiller/workspace
PROJ_ROOT=/Users/distiller/project
export TCLLIBPATH="/usr/local/lib" 
# Install conda
curl -o ~/Downloads/conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
chmod +x ~/Downloads/conda.sh
/bin/bash ~/Downloads/conda.sh -b -p ~/anaconda
export PATH="~/anaconda/bin:${PATH}"
source ~/anaconda/bin/activate
# Install dependencies
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing requests
# sync submodules
cd ${PROJ_ROOT}
git submodule sync
git submodule update --init --recursive
# export 
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# run build script
chmod a+x ${PROJ_ROOT}/scripts/build_ios.sh
echo "########################################################"
cat ${PROJ_ROOT}/scripts/build_ios.sh
echo "########################################################"
echo "IOS_ARCH: ${IOS_ARCH}"
echo "IOS_PLATFORM: ${IOS_PLATFORM}"
BUILD_PYTORCH_MOBILE=1
# unbuffer ${PROJ_ROOT}/scripts/build_ios.sh 2>&1 | ts
CMAKE_ARGS=() 
if [ -n "${IOS_ARCH:-}" ]; then
  CMAKE_ARGS+=("-DIOS_ARCH=${IOS_ARCH}")
  export IOS_ARCH=${IOS_ARCH}
fi
if [ -n "${IOS_PLATFORM:-}" ]; then
  CMAKE_ARGS+=("-DIOS_PLATFORM=${IOS_PLATFORM}")
fi
if [ -n "${USE_NNPACK:-}" ]; then 
  CMAKE_ARGS+=("-DUSE_NNPACK=${USE_NNPACK}")
fi 
CMAKE_ARGS+=("-DBUILD_CAFFE2_MOBILE=OFF")
CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')")
CMAKE_ARGS+=("-DPYTHON_EXECUTABLE=$(python -c 'import sys; print(sys.executable)')")
unbuffer ${PROJ_ROOT}/scripts/build_ios.sh ${CMAKE_ARGS[@]} 2>&1 | ts
# archive the static libraries
cd ${PROJ_ROOT}/build_ios/install/lib
x86_libs=(libc10.a libclog.a libcpuinfo.a libqnnpack.a libtorch.a)
arm_libs=(libc10.a libclog.a libcpuinfo.a libnnpack.a libqnnpack.a libtorch.a)
OUTPUT=libtorch_${IOS_ARCH}.a
cd ${PROJ_ROOT}/build_ios/install/lib
if [ "${IOS_PLATFORM}" == "SIMULATOR" ]; then
  libtool -static x86_libs[@] -o ${OUTPUT}
else
  libtool -static arm_libs[@] -o ${OUTPUT}
fi
#store the binary
cd ${WORKSPACE}
DEST_DIR=${WORKSPACE}/ios
mkdir -p ${DEST_DIR}
cp -R ${PROJ_ROOT}/build_ios/install ${DEST_DIR}
mv ${DEST_DIR}/install ${DEST_DIR}/${IOS_ARCH}