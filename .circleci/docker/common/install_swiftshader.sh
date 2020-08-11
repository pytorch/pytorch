#!/bin/bash

set -ex

[ -n "${SWIFTSHADER}" ]

retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

_https_amazon_aws=https://ossci-android.s3.amazonaws.com

apt-get update
sudo apt-get install -y libx11-dev libxcb1-dev libx11-xcb-dev libxext-dev libegl-mesa0 libglfw3-dev libgles2-mesa-dev

# CMake >= 3.13 is required by SwiftShader
CMAKE_VERSION_SWIFTSHADER=3.16.8
# Turn 3.16.8 into v3.16
_cmake_vpath=$(echo "${CMAKE_VERSION_SWIFTSHADER}" | sed -e 's/\([0-9].[0-9]\+\).*/v\1/')
_cmake_file="cmake-${CMAKE_VERSION_SWIFTSHADER}-Linux-x86_64"

_cmake_dir=/var/lib/jenkins/swiftshader-cmake
mkdir -p $_cmake_dir
_tmp_cmake_targz="/tmp/cmake-${CMAKE_VERSION_SWIFTSHADER}.tar.gz"

curl --silent --show-error --location --fail --retry 3 \
  --output "${_tmp_cmake_targz}" "https://cmake.org/files/${_cmake_vpath}/${_cmake_file}.tar.gz"

tar -C "${_cmake_dir}" -xzf "${_tmp_cmake_targz}"
_cmake_bin_path="${_cmake_dir}/${_cmake_file}/bin/cmake"
rm "${_tmp_cmake_targz}"

echo "XXX-install-swiftshader-env"
env
echo "XXX-install-swiftshader-env~"

# SwiftShader
_swiftshader_root_dir=/var/lib/jenkins
_swiftshader_dir="${_swiftshader_root_dir}"/swiftshader
retry git clone https://github.com/google/swiftshader.git "${_swiftshader_dir}"
pushd "${_swiftshader_dir}"
git submodule sync && git submodule update -q --init --recursive
popd

pushd "${_swiftshader_dir}"/build

$_cmake_bin_path \
  -DCMAKE_BUILD_TYPE=Release \
  -DSWIFTSHADER_BUILD_VULKAN=1 \
  -DSWIFTSHADER_BUILD_EGL=1 \
  -DSWIFTSHADER_BUILD_GLESv2=1 \
  -DSWIFTSHADER_BUILD_GLES_CM=1 \
  -DSWIFTSHADER_BUILD_PVR=0 \
  -DSWIFTSHADER_BUILD_TESTS=1 \
  -DSWIFTSHADER_WARNINGS_AS_ERRORS=1 \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_AR=/usr/bin/llvm-ar-"$CLANG_VERSION" \
  -DCMAKE_RANLIB=/usr/bin/llvm-ranlib-"$CLANG_VERSION" \
  ..

MAX_JOBS=$(nproc)
make SHELL='sh -x' VERBOSE=1 AM_DEFAULT_VERBOSITY=1 --debug=j --jobs=${MAX_JOBS}

./vk-unittests

popd

export VK_ICD_FILENAMES="${_swiftshader_dir}/build/Linux/vk_swiftshader_icd.json"
