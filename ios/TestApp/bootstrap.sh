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
    echo "detecting devices..."
    if ! [ -x "$(command -v ios-deploy)" ]; then
        echo 'Error: ios-deploy is not installed.'
        exit 1
    fi 
    ios-deploy -c -t 1
    if [ "$?" -ne "0" ]; then
        echo 'Error: No device connected. Please connect your device via USB then re-run the script'
        exit 1
    fi
    echo "Done."
    PROJ_ROOT=$(pwd)
    BENCHMARK_DIR="${PROJ_ROOT}/benchmark"
    XCODE_PROJ_PATH="./TestApp.xcodeproj"
    XCODE_TARGET="TestApp"
    XCODE_BUILD="./build"
    if [ ! -f "./.config" ]; then 
        touch .config
        echo "" >> .config
    else
        source .config
    fi
    if [ -z "${TEAM_ID}" ]; then 
        reply=$(bash -c 'read -r -p "Team Id:" tmp; echo $tmp')
        TEAM_ID="${reply}"
        echo "TEAM_ID=${TEAM_ID}" >> .config
    fi
    if [ -z "${PROFILE}" ]; then 
        reply=$(bash -c 'read -r -p "Provisioning Profile:" tmp; echo $tmp')
        PROFILE="${reply}"
        echo "PROFILE=${PROFILE}" >> .config
    fi
    if [ -d "${XCODE_BUILD}" ]; then
        echo "found the old XCode build, remove it"
        rm -rf "${XCODE_BUILD}"
    fi 
    cd "${BENCHMARK_DIR}"
    echo "Generating model"
    python trace_model.py
    ruby setup.rb -t "${TEAM_ID}"
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
    echo "installing..."
    ios-deploy -r --bundle "${XCODE_BUILD}/Debug-iphoneos/${XCODE_TARGET}.app"
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

bootstrap
