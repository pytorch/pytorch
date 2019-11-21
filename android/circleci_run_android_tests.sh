#!/bin/bash
set -eux -o pipefail 

ANDROID_SDK=/home/ivankobzarev/android_sdk/r19c
GRADLE_PATH=/home/ivankobzarev/gradle/gradle-5.4.1/bin/gradle
ADB_PATH=$ANDROID_SDK/platform-tools/adb
PYTORCH_ANDROID_DIR=/home/ivankobzarev/pytorch/master/pytorch/android
PROXY_PARAMS="--proxy=http --proxy_host=fwdproxy --proxy_port=8080"
EMULATOR_CONFIG="system-images;android-25;google_apis;x86"
EMULATOR_NAME="test_25_x86"

$ANDROID_SDK/tools/bin/sdkmanager $PROXY_PARAMS "$EMULATOR_CONFIG"

echo "no" | $ANDROID_SDK/tools/bin/avdmanager create avd \
  --force \
  -n "$EMULATOR_NAME" \
  -k "$EMULATOR_CONFIG"

# Checking if there is an active emulator process
ps -ef | grep emulator | grep $EMULATOR_NAME

$ANDROID_SDK/emulator/emulator-headless \
  -avd "$EMULATOR_NAME" \
  -noaudio -no-boot-anim -no-window -accel on&
EMULATOR_PID=$!

echo "Waiting for emulator boot completed"
$ADB_PATH wait-for-device shell 'while [[ -z $(getprop sys.boot_completed) ]]; do sleep 1; done;'
$ADB_PATH devices

$GRADLE_PATH -PABI_FILTERS=x86 -p $PYTORCH_ANDROID_DIR pytorch_android:connectedAndroidTest

kill -9 $EMULATOR_PID
ps -ef | grep emulator | grep $EMULATOR_NAME

