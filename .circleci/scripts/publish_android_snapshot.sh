#!/usr/bin/env bash
# DO NOT ADD 'set -x' not to reveal CircleCI secret context environment variables
set -eu -o pipefail

export ANDROID_NDK_HOME=/opt/ndk
export ANDROID_HOME=/opt/android/sdk

export GRADLE_VERSION=4.10.3
export GRADLE_HOME=/opt/gradle/gradle-$GRADLE_VERSION
export GRADLE_PATH=$GRADLE_HOME/bin/gradle

echo "BUILD_ENVIRONMENT:$BUILD_ENVIRONMENT"
ls -la ~/workspace

GRADLE_PROPERTIES=~/workspace/android/gradle.properties

IS_SNAPSHOT="$(grep 'VERSION_NAME=[0-9\.]\+-SNAPSHOT' "$GRADLE_PROPERTIES")"
echo "IS_SNAPSHOT:$IS_SNAPSHOT"

if [ -z "$IS_SNAPSHOT" ]; then
  echo "Error: version is not snapshot."
elif [ -z "$SONATYPE_NEXUS_USERNAME" ]; then
  echo "Error: missing env variable SONATYPE_NEXUS_USERNAME."
elif [ -z "$SONATYPE_NEXUS_PASSWORD" ]; then
  echo "Error: missing env variable SONATYPE_NEXUS_PASSWORD."
elif [ -z "$ANDROID_SIGN_KEY" ]; then
  echo "Error: missing env variable ANDROID_SIGN_KEY."
elif [ -z "$ANDROID_SIGN_PASS" ]; then
  echo "Error: missing env variable ANDROID_SIGN_PASS."
else
  GRADLE_LOCAL_PROPERTIES=~/workspace/android/local.properties
  rm -f $GRADLE_LOCAL_PROPERTIES

  echo "sdk.dir=/opt/android/sdk" >> $GRADLE_LOCAL_PROPERTIES
  echo "ndk.dir=/opt/ndk" >> $GRADLE_LOCAL_PROPERTIES

  echo "SONATYPE_NEXUS_USERNAME=${SONATYPE_NEXUS_USERNAME}" >> $GRADLE_PROPERTIES
  echo "SONATYPE_NEXUS_PASSWORD=${SONATYPE_NEXUS_PASSWORD}" >> $GRADLE_PROPERTIES

  echo "signing.keyId=${ANDROID_SIGN_KEY}" >> $GRADLE_PROPERTIES
  echo "signing.password=${ANDROID_SIGN_PASS}" >> $GRADLE_PROPERTIES

  $GRADLE_PATH -p ~/workspace/android/ uploadArchives
fi
