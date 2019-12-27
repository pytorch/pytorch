#!/bin/bash
set -eux

PYTORCH_DIR="$(cd $(dirname $0)/..; pwd -P)"
PYTORCH_ANDROID_DIR=$PYTORCH_DIR/android

echo "ANDROID_HOME:$ANDROID_HOME"
if [ ! -z "$ANDROID_HOME" ]; then
  echo "ANDROID_HOME not set; please set it to Android sdk directory"
fi

if [ ! -d $ANDROID_HOME ]; then
  echo "ANDROID_HOME not a directory; did you install it under $ANDROID_HOME?"
  exit 1
fi

echo "ANDROID_NDK:$ANDROID_NDK"
if [ ! -z "$ANDROID_NDK" ]; then
  echo "ANDROID_NDK not set; please set it to Android sdk directory"
fi

if [ ! -d $ANDROID_NDK ]; then
  echo "ANDROID_NDK not a directory; did you install it under $ANDROID_NDK?"
  exit 1
fi

GRADLE_PATH=gradle
GRADLE_NOT_FOUND_MSG="Unable to find gradle, please add it to PATH or set GRADLE_HOME"

if [ ! -x "$(command -v gradle)" ]; then
  if [ -z "$GRADLE_HOME" ]; then
    echo GRADLE_NOT_FOUND_MSG 
    exit 1
  fi
  GRADLE_PATH=$GRADLE_HOME/bin/gradle
  if [ ! -f "$GRADLE_PATH" ]; then
    echo GRADLE_NOT_FOUND_MSG 
    exit 1
  fi
fi
echo "GRADLE_PATH:$GRADLE_PATH"


# Run android instrumented tests on x86 emulator

ADB_PATH=$ANDROID_HOME/platform-tools/adb

echo "Expecting running emulator"
$ADB_PATH devices

DEVICES_COUNT=$($ADB_PATH devices | awk 'NF' | wc -l)
echo "DEVICES_COUNT:$DEVICES_COUNT"

if [ "$DEVICES_COUNT" -eq 1 ]; then
  echo "Unable to found connected android emulators"
cat <<- EOF
  To start android emulator:
  1. Install android sdkmanager packages
  $ANDROID_HOME/tools/bin/sdkmanager "system-images;android-25;google_apis;x86"

  to specify proxy add params: --proxy=http --proxy_host=fwdproxy --proxy_port=8080

  2. Create android virtual device
  $ANDROID_HOME/tools/bin/avdmanager create avd --name "x86_android25" --package "system-images;android-25;google_apis;x86"

  3. Start emulator in headless mode without audio
  $ANDROID_HOME/tools/emulator -avd x86_android25 -no-audio -no-window

  4. Check that emulator is running
  $ANDROID_HOME/platform-tools/adb devices

  If everything is ok the output will be:
  
  List of devices attached
  emulator-5554   device
EOF
  exit 1
fi

echo "Waiting for emulator boot completed"
$ADB_PATH wait-for-device shell 'while [[ -z $(getprop sys.boot_completed) ]]; do sleep 1; done;'

$GRADLE_PATH -PABI_FILTERS=x86 -p $PYTORCH_ANDROID_DIR connectedAndroidTest
