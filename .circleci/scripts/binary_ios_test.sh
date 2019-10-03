#!/bin/bash

echo ""
echo "PWD: $(pwd)"

echo "IOS_ARCH: ${IOS_ARCH}"
echo "IOS_PLATFORM: ${IOS_PLATFORM}"
export IOS_ARCH=${IOS_ARCH}
export IOS_PLATFORM=${IOS_PLATFORM}
WORKSPACE=/Users/distiller/workspace
PROJ_ROOT=/Users/distiller/project

if [ ${IOS_PLATFORM} = "SIMULAOTR" ]; then 
    ruby ${PROJ_ROOT}/scripts/xcode_ios_x86_build.rb -i ${PROJ_ROOT}/build_ios/install/lib -x ${PROJ_ROOT}/ios/TestApp/TestApp.xcodeproj
fi