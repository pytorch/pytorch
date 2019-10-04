#!/bin/bash

echo ""
echo "PWD: $(pwd)"
echo "IOS_PLATFORM: ${IOS_PLATFORM}"
PROJ_ROOT=/Users/distiller/project
if ! [ -x "$(command -v xcodebuild)" ]; then
    echo 'Error: xcodebuild is not installed.'
    exit 1
fi 
ruby ${PROJ_ROOT}/scripts/xcode_build.rb -i ${PROJ_ROOT}/build_ios/install -x ${PROJ_ROOT}/ios/TestApp/TestApp.xcodeproj -p ${IOS_PLATFORM}
if ! [ "$?" -eq "0" ]; then
    echo 'xcodebuild failed!'
    exit 1
fi