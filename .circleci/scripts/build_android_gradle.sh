#!/usr/bin/env bash
set -eux -o pipefail

export ANDROID_NDK_HOME=/opt/ndk
export ANDROID_HOME=/opt/android/sdk

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

export GRADLE_LOCAL_PROPERTIES=~/workspace/android/local.properties
rm -f $GRADLE_LOCAL_PROPERTIES
echo "sdk.dir=/opt/android/sdk" >> $GRADLE_LOCAL_PROPERTIES
echo "ndk.dir=/opt/ndk" >> $GRADLE_LOCAL_PROPERTIES

if [[ "${BUILD_ENVIRONMENT}" == *-gradle-build-only-x86_32* ]]; then
    $GRADLE_PATH -PABI_FILTERS=x86 -p ~/workspace/android/ assembleRelease
else
    $GRADLE_PATH -p ~/workspace/android/ assembleRelease
fi

find . -type f -name *aar -print | xargs tar cfvz ~/workspace/android/artifacts.tgz
