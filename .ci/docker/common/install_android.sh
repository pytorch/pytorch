#!/bin/bash

set -ex

[ -n "${ANDROID_NDK}" ]

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

sudo -H -u jenkins $GRADLE_HOME/bin/gradle -Pandroid.useAndroidX=true -p /var/lib/jenkins/gradledeps -g /var/lib/jenkins/.gradle --refresh-dependencies --debug --stacktrace assemble

chown -R jenkins /var/lib/jenkins/.gradle
chgrp -R jenkins /var/lib/jenkins/.gradle

popd

rm -rf /var/lib/jenkins/.gradle/daemon

# Cache vision models used by the test
source "$(dirname "${BASH_SOURCE[0]}")/cache_vision_models.sh"
