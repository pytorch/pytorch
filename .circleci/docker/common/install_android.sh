#!/bin/bash

set -ex

[ -n "${ANDROID_NDK}" ]

retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

_https_amazon_aws=https://ossci-android.s3.amazonaws.com

apt-get update
apt-get install -y --no-install-recommends autotools-dev autoconf unzip
apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

pushd /tmp
curl -Os --retry 3 $_https_amazon_aws/android-ndk-${ANDROID_NDK}-linux-x86_64.zip
popd
_ndk_dir=/opt/ndk
mkdir -p "$_ndk_dir"
unzip -qo /tmp/android*.zip -d "$_ndk_dir"
_versioned_dir=$(find "$_ndk_dir/" -mindepth 1 -maxdepth 1 -type d)
mv "$_versioned_dir"/* "$_ndk_dir"/
rmdir "$_versioned_dir"
rm -rf /tmp/*

# Install OpenJDK
# https://hub.docker.com/r/picoded/ubuntu-openjdk-8-jdk/dockerfile/

sudo apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/oracle-jdk8-installer;

# Fix certificate issues, found as of
# https://bugs.launchpad.net/ubuntu/+source/ca-certificates-java/+bug/983302

sudo apt-get update && \
    apt-get install -y ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/oracle-jdk8-installer;

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/

# Installing android sdk
# https://github.com/circleci/circleci-images/blob/staging/android/Dockerfile.m4

_tmp_sdk_zip=/tmp/android-sdk-linux.zip
_android_home=/opt/android/sdk

rm -rf $_android_home
sudo mkdir -p $_android_home
curl --silent --show-error --location --fail --retry 3 --output /tmp/android-sdk-linux.zip $_https_amazon_aws/android-sdk-linux-tools3859397-build-tools2803-2902-platforms28-29.zip
sudo unzip -q $_tmp_sdk_zip -d $_android_home
rm $_tmp_sdk_zip

sudo chmod -R 777 $_android_home

export ANDROID_HOME=$_android_home
export ADB_INSTALL_TIMEOUT=120

export PATH="${ANDROID_HOME}/tools:${ANDROID_HOME}/tools/bin:${ANDROID_HOME}/platform-tools:${PATH}"
echo "PATH:${PATH}"

# Installing Vulkan Sdk
_vulkansdk_dir=/var/lib/jenkins/vulkansdk
mkdir -p $_vulkansdk_dir
_tmp_vulkansdk_targz=/tmp/vulkansdk.tar.gz
curl --silent --show-error --location --fail --retry 3 --output "$_tmp_vulkansdk_targz" "$_https_amazon_aws/vulkansdk-linux-x86_64-1.2.148.0.tar.gz"
tar -C "$_vulkansdk_dir" -xzf "$_tmp_vulkansdk_targz"
export VULKAN_SDK="$_vulkansdk_dir/1.2.148.0/"
rm "$_tmp_vulkansdk_targz"


# Installing SwiftShader for Vulkan
# XCB libs
apt-get update
sudo apt-get install -y libx11-dev libxcb1-dev libx11-xcb-dev

# CMake >= 3.13 is required by SwiftShader
_cmake_dir=/var/lib/jenkins/swiftshader-cmake
mkdir -p $_cmake_dir
_tmp_cmake_targz=/tmp/cmake.tar.gz
curl --silent --show-error --location --fail --retry 3 --output $_tmp_cmake_targz https://cmake.org/files/v3.16/cmake-3.16.8-Linux-x86_64.tar.gz
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

$_cmake_bin_path ..
make --jobs=8
./vk-unittests

popd

export VK_ICD_FILENAMES="$_swiftshader_dir/build/Linux/vk_swiftshader_icd.json"

# Installing Gradle
echo "GRADLE_VERSION:${GRADLE_VERSION}"
_gradle_home=/opt/gradle
sudo rm -rf $gradle_home
sudo mkdir -p $_gradle_home

curl --silent --output /tmp/gradle.zip --retry 3 $_https_amazon_aws/gradle-${GRADLE_VERSION}-bin.zip

sudo unzip -q /tmp/gradle.zip -d $_gradle_home
rm /tmp/gradle.zip

sudo chmod -R 777 $_gradle_home

export GRADLE_HOME=$_gradle_home/gradle-$GRADLE_VERSION
alias gradle="${GRADLE_HOME}/bin/gradle"

export PATH="${GRADLE_HOME}/bin/:${PATH}"
echo "PATH:${PATH}"

gradle --version

mkdir /var/lib/jenkins/gradledeps
cp build.gradle /var/lib/jenkins/gradledeps
cp AndroidManifest.xml /var/lib/jenkins/gradledeps

pushd /var/lib/jenkins

export GRADLE_LOCAL_PROPERTIES=gradledeps/local.properties
rm -f $GRADLE_LOCAL_PROPERTIES
echo "sdk.dir=/opt/android/sdk" >> $GRADLE_LOCAL_PROPERTIES
echo "ndk.dir=/opt/ndk" >> $GRADLE_LOCAL_PROPERTIES

chown -R jenkins /var/lib/jenkins/gradledeps
chgrp -R jenkins /var/lib/jenkins/gradledeps

sudo -H -u jenkins $GRADLE_HOME/bin/gradle -p /var/lib/jenkins/gradledeps -g /var/lib/jenkins/.gradle --refresh-dependencies --debug --stacktrace assemble

chown -R jenkins /var/lib/jenkins/.gradle
chgrp -R jenkins /var/lib/jenkins/.gradle

popd

rm -rf /var/lib/jenkins/.gradle/daemon
