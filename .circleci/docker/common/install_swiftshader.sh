#!/bin/bash

set -ex

[ -n "${SWIFTSHADER}" ]

retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

_https_amazon_aws=https://ossci-android.s3.amazonaws.com

apt-get update
sudo apt-get install -y libx11-dev libxcb1-dev libx11-xcb-dev

# CMake >= 3.13 is required by SwiftShader
_cmake_dir=/var/lib/jenkins/swiftshader-cmake
mkdir -p $_cmake_dir
_tmp_cmake_targz=/tmp/cmake.tar.gz
curl --silent --show-error --location --fail --retry 3 \
  --output $_tmp_cmake_targz https://cmake.org/files/v3.16/cmake-3.16.8-Linux-x86_64.tar.gz

tar -C "$_cmake_dir" -xzf "$_tmp_cmake_targz"
_cmake_bin_path="$_cmake_dir/cmake-3.16.8-Linux-x86_64/bin/cmake"
rm "$_tmp_cmake_targz"

# SwiftShader
_swiftshader_root_dir=/var/lib/jenkins
_swiftshader_dir="$_swiftshader_root_dir/swiftshader"
retry git clone https://github.com/google/swiftshader.git "$_swiftshader_dir"
pushd "$_swiftshader_dir"
git submodule sync && git submodule update -q --init --recursive
popd
#_tmp_swiftshader_zip=/tmp/swiftshader-master-200805-1128.zip
#curl --silent --show-error --location --fail --retry 3 --output "$_tmp_swiftshader_zip" "$_https_amazon_aws/swiftshader-master-200805-1128.zip"
#unzip -qo "$_tmp_swiftshader_zip" -d "$_swiftshader_root_dir"
#_swiftshader_dir="$_swiftshader_root_dir/swiftshader-master"
#rm "$_tmp_swiftshader_zip"

pushd "$_swiftshader_dir/build"

$_cmake_bin_path \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DSWIFTSHADER_BUILD_VULKAN=1 \
  -DSWIFTSHADER_BUILD_EGL=0 \
  -DSWIFTSHADER_BUILD_GLESv2=0 \
  -DSWIFTSHADER_BUILD_GLES_CM=0 \
  -DSWIFTSHADER_BUILD_PVR=0 \
  -DSWIFTSHADER_BUILD_TESTS=0 \
  -DSWIFTSHADER_LESS_DEBUG_INFO=1 \
  -DSWIFTSHADER_WARNINGS_AS_ERRORS=1 \
  -D_CMAKE_TOOLCHAIN_PREFIX=llvm- \
  -DCMAKE_EXE_LINKER_FLAGS=" -fuse-ld=gold " \
  ..

make SHELL='sh -x' VERBOSE=1 AM_DEFAULT_VERBOSITY=1 --debug=j --jobs=8

./vk-unittests

popd

export VK_ICD_FILENAMES="$_swiftshader_dir/build/Linux/vk_swiftshader_icd.json"
