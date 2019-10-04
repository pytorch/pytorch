#!/bin/bash

echo ""
echo "PWD: $(pwd)"
echo "IOS_ARCH: ${IOS_ARCH}"
echo "IOS_PLATFORM: ${IOS_PLATFORM}"
PROJ_ROOT=/Users/distiller/project
if ! [ -x "$(command -v xcodebuild)" ]; then
    echo 'Error: xcodebuild is not installed.'
    exit 1
fi 
if [ ${IOS_PLATFORM} = "SIMULAOTR" ]; then 
    ruby ${PROJ_ROOT}/scripts/xcode_ios_x86_build.rb -i ${PROJ_ROOT}/build_ios/install/lib -x ${PROJ_ROOT}/ios/TestApp/TestApp.xcodeproj
    if [ "$?" -eq "0" ]; then
        echo 'xcodebuild failed!'
        exit 1
    fi
fi