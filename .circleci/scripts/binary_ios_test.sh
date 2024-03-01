#!/bin/bash
set -ex -o pipefail

if ! [ "$IOS_PLATFORM" == "SIMULATOR" ]; then
    exit 0
fi

echo ""
echo "DIR: $(pwd)"
PROJ_ROOT=/Users/distiller/project
cd ${PROJ_ROOT}/ios/TestApp
# install fastlane
sudo gem install bundler && bundle install
# run the ruby build script
if ! [ -x "$(command -v xcodebuild)" ]; then
    echo 'Error: xcodebuild is not installed.'
    exit 1
fi
ruby ${PROJ_ROOT}/scripts/xcode_build.rb -i ${PROJ_ROOT}/build_ios/install -x ${PROJ_ROOT}/ios/TestApp/TestApp.xcodeproj -p ${IOS_PLATFORM}
