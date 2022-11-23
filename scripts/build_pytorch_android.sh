#!/bin/bash
set -eux

##############################################################################
# Master script to build PyTorch Android library with Java bindings.
##############################################################################
# Example usage:
# - Build default AARs:
#   scripts/build_pytorch_android.sh
#
# - Build for specific ABI(s):
#   scripts/build_pytorch_android.sh armeabi-v7a
#   scripts/build_pytorch_android.sh arm64-v8a,x86,x86_64
#
# Script's workflow:
# 1. Builds libtorch for android for specified android abisi (by default for all 4).
# Custom list of android abis can be specified as a bash argument as comma separated list.
# For example just for testing on android x86 emulator we need only x86 build.
# ./scripts/build_pytorch_android.sh x86
# 2. Creates symbolic links to android/pytorch_android/src/main/jniLibs/${abi} for libtorch build output,
# android/pytorch_android/src/main/cpp/libtorch_include/${abi} for headers.
# 3. Runs pyotrch_android gradle build:
# gradle assembleRelease

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
  # Skipping clean task here as android gradle plugin 3.3.2 exteralNativeBuild has problems
  # with it when abiFilters are specified.
  $GRADLE_PATH -PABI_FILTERS=$ABIS_LIST -p $PYTORCH_ANDROID_DIR assembleRelease
else
  $GRADLE_PATH -p $PYTORCH_ANDROID_DIR clean assembleRelease
fi

find $PYTORCH_ANDROID_DIR -type f -name *aar | xargs ls -lah
