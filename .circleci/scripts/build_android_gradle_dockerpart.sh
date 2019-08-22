#!/usr/bin/env bash
set -eux -o pipefail

echo "$(pwd)"
echo "HOME:$(~/.)"

# ---------------------------------
# Installing openjdk-8
# https://hub.docker.com/r/picoded/ubuntu-openjdk-8-jdk/dockerfile/

sudo apt-get update && \
    sudo apt-get install -y openjdk-8-jdk && \
    sudo apt-get install -y ant && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/* && \
    sudo rm -rf /var/cache/oracle-jdk8-installer;

sudo apt-get update && \
    sudo apt-get install -y ca-certificates-java && \
    sudo apt-get clean && \
    sudo update-ca-certificates -f && \
    sudo rm -rf /var/lib/apt/lists/* && \
    sudo rm -rf /var/cache/oracle-jdk8-installer;

# ---------------------------------
# Installing android sdk
# https://github.com/circleci/circleci-images/blob/staging/android/Dockerfile.m4

_sdk_version=sdk-tools-linux-3859397.zip
_android_home=/opt/android/sdk

rm -rf $_android_home
sudo mkdir -p $_android_home
curl --silent --show-error --location --fail --retry 3 --output /tmp/$_sdk_version https://dl.google.com/android/repository/$_sdk_version
sudo unzip -q /tmp/$_sdk_version -d $_android_home
rm /tmp/$_sdk_version

sudo chmod -R 777 $_android_home

export ANDROID_HOME=$_android_home
export ADB_INSTALL_TIMEOUT=120

export PATH="${ANDROID_HOME}/emulator:${ANDROID_HOME}/tools:${ANDROID_HOME}/tools/bin:${ANDROID_HOME}/platform-tools:${PATH}"
echo "PATH:${PATH}"

alias sdkmanager=${ANDROID_HOME}/tools/bin/sdkmanager

sudo mkdir /var/lib/jenkins/.android/
sudo chmod -R 777 /var/lib/jenkins/.android/
echo '### User Sources for Android SDK Manager' > /var/lib/jenkins/.android/repositories.cfg
sudo chmod -R 777 /var/lib/jenkins/.android/

sudo yes | sudo sdkmanager --update
sudo yes | sudo sdkmanager --licenses

sdkmanager \
  "tools" \
  "platform-tools" \
  "emulator"

sdkmanager \
  "build-tools;28.0.3"

sdkmanager "platforms;android-28"

sdkmanager --list

# ---------------------------------
# Installing android sdk
# https://github.com/keeganwitt/docker-gradle/blob/a206b4a26547df6d8b29d06dd706358e3801d4a9/jdk8/Dockerfile
export GRADLE_VERSION=5.1.1
_gradle_home=/opt/gradle
sudo rm -rf $_gradle_home
sudo mkdir -p $_gradle_home

wget --no-verbose --output-document=/tmp/gradle.zip \
"https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip"

sudo unzip -q /tmp/gradle.zip -d $_gradle_home
rm /tmp/gradle.zip

sudo chmod -R 777 $_gradle_home

export GRADLE_HOME=$_gradle_home/gradle-$GRADLE_VERSION

export PATH="${GRADLE_HOME}/bin/:${PATH}"
echo "PATH:${PATH}"

alias gradle=$GRADLE_HOME/bin/gradle

gradle --version
