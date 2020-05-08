#!/bin/bash
set -eux

PYTORCH_DIR="$(cd $(dirname $0)/..; pwd -P)"

PYTORCH_ANDROID_DIR=$PYTORCH_DIR/android
WORK_DIR=$PYTORCH_DIR

echo "PYTORCH_DIR:$PYTORCH_DIR"
echo "WORK_DIR:$WORK_DIR"

echo "ANDROID_HOME:$ANDROID_HOME"
if [ ! -z "$ANDROID_HOME" ]; then
  echo "ANDROID_HOME not set; please set it to Android sdk directory"
fi

if [ ! -d $ANDROID_HOME ]; then
  echo "ANDROID_HOME not a directory; did you install it under $ANDROID_HOME?"
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

ABIS_LIST="armeabi-v7a,arm64-v8a,x86,x86_64"
CUSTOM_ABIS_LIST=false
if [ $# -gt 0 ]; then
  ABIS_LIST=$1
  CUSTOM_ABIS_LIST=true
fi

echo "ABIS_LIST:$ABIS_LIST"

LIB_DIR=$PYTORCH_ANDROID_DIR/pytorch_android/src/main/jniLibs
INCLUDE_DIR=$PYTORCH_ANDROID_DIR/pytorch_android/src/main/cpp/libtorch_include
mkdir -p $LIB_DIR
rm -f $LIB_DIR/*
mkdir -p $INCLUDE_DIR

for abi in $(echo $ABIS_LIST | tr ',' '\n')
do
echo "abi:$abi"

OUT_DIR=$WORK_DIR/build_android_$abi

rm -rf $OUT_DIR
mkdir -p $OUT_DIR

pushd $PYTORCH_DIR
python $PYTORCH_DIR/setup.py clean

ANDROID_ABI=$abi VERBOSE=1 ANDROID_DEBUG_SYMBOLS=1 $PYTORCH_DIR/scripts/build_android.sh -DANDROID_CCACHE=$(which ccache)

cp -R $PYTORCH_DIR/build_android/install/lib $OUT_DIR/
cp -R $PYTORCH_DIR/build_android/install/include $OUT_DIR/

echo "$abi build output lib,include copied to $OUT_DIR"

LIB_LINK_PATH=$LIB_DIR/$abi
INCLUDE_LINK_PATH=$INCLUDE_DIR/$abi

rm -f $LIB_LINK_PATH
rm -f $INCLUDE_LINK_PATH

ln -s $OUT_DIR/lib $LIB_LINK_PATH
ln -s $OUT_DIR/include $INCLUDE_LINK_PATH

done

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

popd
