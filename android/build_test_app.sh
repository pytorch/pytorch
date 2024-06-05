#!/bin/bash
set -eux

PYTORCH_DIR="$(cd $(dirname $0)/..; pwd -P)"
PYTORCH_ANDROID_DIR=$PYTORCH_DIR/android

echo "PYTORCH_DIR:$PYTORCH_DIR"

source "$PYTORCH_ANDROID_DIR/common.sh"

check_android_sdk
check_gradle
parse_abis_list "$@"
build_android

# To set proxy for gradle add following lines to ./gradle/gradle.properties:
# systemProp.http.proxyHost=...
# systemProp.http.proxyPort=8080
# systemProp.https.proxyHost=...
# systemProp.https.proxyPort=8080

if [ "$CUSTOM_ABIS_LIST" = true ]; then
  NDK_DEBUG=1 $GRADLE_PATH -PnativeLibsDoNotStrip=true -PABI_FILTERS=$ABIS_LIST -p $PYTORCH_ANDROID_DIR clean test_app:assembleDebug
else
  NDK_DEBUG=1 $GRADLE_PATH -PnativeLibsDoNotStrip=true -p $PYTORCH_ANDROID_DIR clean test_app:assembleDebug
fi

find $PYTORCH_ANDROID_DIR -type f -name *apk

find $PYTORCH_ANDROID_DIR -type f -name *apk | xargs echo "To install apk run: $ANDROID_HOME/platform-tools/adb install -r "
