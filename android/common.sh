#!/bin/bash
set -eux

##############################################################################
# Common util functions for Android build scripts.
##############################################################################

if [ -z "$PYTORCH_DIR" ]; then
  echo "PYTORCH_DIR not set!"
  exit 1
fi

check_android_sdk() {
  if [ -z "$ANDROID_HOME" ]; then
    echo "ANDROID_HOME not set; please set it to Android sdk directory"
    exit 1
  fi

  if [ ! -d "$ANDROID_HOME" ]; then
    echo "ANDROID_HOME not a directory; did you install it under $ANDROID_HOME?"
    exit 1
  fi
  echo "ANDROID_HOME:$ANDROID_HOME"
}

check_gradle() {
  GRADLE_PATH=$PYTORCH_DIR/android/gradlew
  echo "GRADLE_PATH:$GRADLE_PATH"
}

parse_abis_list() {
  ABIS_LIST="armeabi-v7a,arm64-v8a,x86,x86_64"
  CUSTOM_ABIS_LIST=false
  if [ $# -gt 0 ]; then
    ABIS_LIST=$1
    CUSTOM_ABIS_LIST=true
  fi

  echo "ABIS_LIST:$ABIS_LIST"
  echo "CUSTOM_ABIS_LIST:$CUSTOM_ABIS_LIST"
}

build_android() {
  PYTORCH_ANDROID_DIR="$PYTORCH_DIR/android"
  BUILD_ROOT="${BUILD_ROOT:-$PYTORCH_DIR}"
  echo "BUILD_ROOT:$BUILD_ROOT"

  LIB_DIR="$PYTORCH_ANDROID_DIR/pytorch_android/src/main/jniLibs"
  INCLUDE_DIR="$PYTORCH_ANDROID_DIR/pytorch_android/src/main/cpp/libtorch_include"

  # These directories only contain symbolic links.
  rm -rf "$LIB_DIR" && mkdir -p "$LIB_DIR"
  rm -rf "$INCLUDE_DIR" && mkdir -p "$INCLUDE_DIR"

  for abi in $(echo "$ABIS_LIST" | tr ',' '\n')
  do
    echo "abi:$abi"
    ANDROID_BUILD_ROOT="$BUILD_ROOT/build_android_$abi"
    ANDROID_ABI="$abi" \
      BUILD_ROOT="$ANDROID_BUILD_ROOT" \
      "$PYTORCH_DIR/scripts/build_android.sh" \
      -DANDROID_CCACHE="$(which ccache)"

    echo "$abi build output lib,include at $ANDROID_BUILD_ROOT/install"
    ln -s "$ANDROID_BUILD_ROOT/install/lib" "$LIB_DIR/$abi"
    ln -s "$ANDROID_BUILD_ROOT/install/include" "$INCLUDE_DIR/$abi"
  done
}
