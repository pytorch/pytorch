#!/bin/bash

set -ex

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=./common-build.sh
source "$(dirname "${BASH_SOURCE[0]}")/common-build.sh"
POSSIBLE_JAVA_HOMES=()
POSSIBLE_JAVA_HOMES+=(/usr/local)
POSSIBLE_JAVA_HOMES+=(/usr/lib/jvm/java-8-openjdk-amd64)
POSSIBLE_JAVA_HOMES+=(/Library/Java/JavaVirtualMachines/*.jdk/Contents/Home)
# Add the Windows-specific JNI
POSSIBLE_JAVA_HOMES+=("$PWD/.circleci/windows-jni/")
for JH in "${POSSIBLE_JAVA_HOMES[@]}" ; do
if [[ -e "$JH/include/jni.h" ]] ; then
    # Skip if we're not on Windows but haven't found a JAVA_HOME
    if [[ "$JH" == "$PWD/.circleci/windows-jni/" && "$OSTYPE" != "msys" ]] ; then
    break
    fi
    echo "Found jni.h under $JH"
    export JAVA_HOME="$JH"
    export BUILD_JNI=ON
    break
fi
done
if [ -z "$JAVA_HOME" ]; then
echo "Did not find jni.h"
fi

# Test no-Python build
echo "Building libtorch"
# NB: Install outside of source directory (at the same level as the root
# pytorch folder) so that it doesn't get cleaned away prior to docker push.
BUILD_LIBTORCH_PY=$PWD/tools/build_libtorch.py
mkdir -p ../cpp-build/caffe2
pushd ../cpp-build/caffe2
WERROR=1 VERBOSE=1 DEBUG=1 python "$BUILD_LIBTORCH_PY"
popd
