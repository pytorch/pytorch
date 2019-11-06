#!/usr/bin/env bash
set -eux -o pipefail

export ANDROID_NDK_HOME=/opt/ndk
export ANDROID_HOME=/opt/android/sdk

# Must be in sync with GRADLE_VERSION in docker image for android
# https://github.com/pietern/pytorch-dockerfiles/blob/master/build.sh#L155
export GRADLE_VERSION=4.10.3
export GRADLE_HOME=/opt/gradle/gradle-$GRADLE_VERSION
export GRADLE_PATH=$GRADLE_HOME/bin/gradle

BUILD_ANDROID_INCLUDE_DIR_x86=~/workspace/build_android/install/include
BUILD_ANDROID_LIB_DIR_x86=~/workspace/build_android/install/lib

BUILD_ANDROID_INCLUDE_DIR_x86_64=~/workspace/build_android_install_x86_64/install/include
BUILD_ANDROID_LIB_DIR_x86_64=~/workspace/build_android_install_x86_64/install/lib

BUILD_ANDROID_INCLUDE_DIR_arm_v7a=~/workspace/build_android_install_arm_v7a/install/include
BUILD_ANDROID_LIB_DIR_arm_v7a=~/workspace/build_android_install_arm_v7a/install/lib

BUILD_ANDROID_INCLUDE_DIR_arm_v8a=~/workspace/build_android_install_arm_v8a/install/include
BUILD_ANDROID_LIB_DIR_arm_v8a=~/workspace/build_android_install_arm_v8a/install/lib

PYTORCH_ANDROID_SRC_MAIN_DIR=~/workspace/android/pytorch_android/src/main

JNI_INCLUDE_DIR=${PYTORCH_ANDROID_SRC_MAIN_DIR}/cpp/libtorch_include
mkdir -p $JNI_INCLUDE_DIR

JNI_LIBS_DIR=${PYTORCH_ANDROID_SRC_MAIN_DIR}/jniLibs
mkdir -p $JNI_LIBS_DIR

ln -s ${BUILD_ANDROID_INCLUDE_DIR_x86} ${JNI_INCLUDE_DIR}/x86
ln -s ${BUILD_ANDROID_LIB_DIR_x86} ${JNI_LIBS_DIR}/x86

if [[ "${BUILD_ENVIRONMENT}" != *-gradle-build-only-x86_32* ]]; then
ln -s ${BUILD_ANDROID_INCLUDE_DIR_x86_64} ${JNI_INCLUDE_DIR}/x86_64
ln -s ${BUILD_ANDROID_LIB_DIR_x86_64} ${JNI_LIBS_DIR}/x86_64

ln -s ${BUILD_ANDROID_INCLUDE_DIR_arm_v7a} ${JNI_INCLUDE_DIR}/armeabi-v7a
ln -s ${BUILD_ANDROID_LIB_DIR_arm_v7a} ${JNI_LIBS_DIR}/armeabi-v7a

ln -s ${BUILD_ANDROID_INCLUDE_DIR_arm_v8a} ${JNI_INCLUDE_DIR}/arm64-v8a
ln -s ${BUILD_ANDROID_LIB_DIR_arm_v8a} ${JNI_LIBS_DIR}/arm64-v8a
fi

env
echo "BUILD_ENVIRONMENT:$BUILD_ENVIRONMENT"

GRADLE_PARAMS="-p android assembleRelease --debug --stacktrace"
if [[ "${BUILD_ENVIRONMENT}" == *-gradle-build-only-x86_32* ]]; then
    GRADLE_PARAMS+=" -PABI_FILTERS=x86"
fi

if [ ! -z "{GRADLE_OFFLINE:-}" ]; then
    GRADLE_PARAMS+=" --offline"
fi

# touch gradle cache files to prevent expiration
while IFS= read -r -d '' file
do
  touch "$file" || true
done < <(find /var/lib/jenkins/.gradle -type f -print0)

env
#[ -f /usr/local/bin/cmake ] && sudo mv -f /usr/local/bin/cmake /usr/local/bin/cmake-disabled-for-gradle || true
#[ -f /usr/bin/cmake ] && sudo mv -f /usr/bin/cmake /usr/bin/cmake-disabled-for-gradle || true
which cmake || true

CMAKE_DIR=/opt/cmake_gradle
sudo mkdir -p "$CMAKE_DIR"
sudo chmod -R 777 "$CMAKE_DIR"

pushd "$CMAKE_DIR"
curl -Os "https://cmake.org/files/v3.7/cmake-3.7.0-Linux-x86_64.tar.gz"
tar -C "$CMAKE_DIR" --strip-components 1 --no-same-owner -zxf cmake-*.tar.gz
rm -f cmake-*.tar.gz
popd

$CMAKE_DIR/bin/cmake --version

export GRADLE_LOCAL_PROPERTIES=~/workspace/android/local.properties
rm -f $GRADLE_LOCAL_PROPERTIES
echo "sdk.dir=/opt/android/sdk" >> $GRADLE_LOCAL_PROPERTIES
echo "ndk.dir=/opt/ndk" >> $GRADLE_LOCAL_PROPERTIES
echo "cmake.dir=$CMAKE_DIR" >> $GRADLE_LOCAL_PROPERTIES

$GRADLE_PATH $GRADLE_PARAMS || true

find . -type f -name "*.a" -exec ls -lh {} \;

while IFS= read -r -d '' file
do
  echo
  echo "$file"
  ls -lah "$file"
  zipinfo -l "$file"
done < <(find . -type f -name '*.aar' -print0)

find . -type f -name *aar -print | xargs tar cfvz ~/workspace/android/artifacts.tgz

