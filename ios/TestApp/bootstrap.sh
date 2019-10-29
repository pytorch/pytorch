#!/bin/bash
set -e

echo "Current Dir: $(pwd)"
if [[ "$OSTYPE" != *"darwin"* ]];then
    error "Current OS Type is not MacOS"
    sleep 1
    exit 1
fi
BIN_NAME=$(basename "$0")
help () {
  echo "Usage: $BIN_NAME <options>"
  echo
  echo "Options:"
  echo "   -t           Team Identifier"
  echo "   -p           Name of the Provisioning Profile"
}
bootstrap() {
    echo "starting"
    PROJ_ROOT=$(pwd)
    BENCHMARK_DIR="${PROJ_ROOT}/benchmark"
    XCODE_PROJ_PATH="./TestApp.xcodeproj"
    XCODE_TARGET="TestApp"
    XCODE_BUILD="./build"
    if [ -d ${XCODE_BUILD} ]; then
        echo "found the old XCode build, remove it"
        rm -rf ${XCODE_BUILD}
    fi 
    cd ${BENCHMARK_DIR}
    echo "Generating model"
    python trace_model.py
    ruby setup.rb -t ${TEAM_ID}
    cd ..
    #run xcodebuild
    if ! [ -x "$(command -v xcodebuild)" ]; then
        echo 'Error: xcodebuild is not installed.'
        exit 1
    fi 
    echo "Running xcodebuild"
    xcodebuild clean build  -project ${XCODE_PROJ_PATH}  \
                            -target ${XCODE_TARGET}  \
                            -sdk iphoneos \
                            -configuration Debug \
                            PROVISIONING_PROFILE_SPECIFIER=${PROFILE}
    #install TestApp
    if ! [ -x "$(command -v ios-deploy)" ]; then
        echo 'Error: ios-deploy is not installed.'
        exit 1
    fi 
    echo "installing, make sure your phone is unlocked"
    ios-deploy --bundle "${XCODE_BUILD}/Debug-iphoneos/${XCODE_TARGET}.app"
    echo "Done."
}
while [[ $# -gt 1 ]]
do
option="$1"
value="$2"
case $option in 
    "" | "-h" | "--help")
    help
    exit 0
    ;;
    "-t" | "--team")
    TEAM_ID="${value}"
    shift
    ;;
    "-p"|"--profile")
    PROFILE="${value}"
    shift
    ;;
    *)
    echo "unknown options" >& 2 
    help
    exit 1
    ;;
esac
shift 
done

echo TEAM_ID = "${TEAM_ID}"
echo PROFILE = "${PROFILE}"

bootstrap
