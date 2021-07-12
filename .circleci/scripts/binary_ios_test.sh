#!/bin/bash
set -ex -o pipefail

echo ""
echo "DIR: $(pwd)"
PROJ_ROOT=/Users/distiller/project
cd ${PROJ_ROOT}/ios/TestApp
# install fastlane
sudo gem install bundler && bundle install
# install certificates
echo "${IOS_CERT_KEY}" >> cert.txt
base64 --decode cert.txt -o Certificates.p12
rm cert.txt
bundle exec fastlane install_cert
# install the provisioning profile
PROFILE=PyTorch_CI_2021.mobileprovision
PROVISIONING_PROFILES=~/Library/MobileDevice/Provisioning\ Profiles
mkdir -pv "${PROVISIONING_PROFILES}"
cd "${PROVISIONING_PROFILES}"
echo "${IOS_SIGN_KEY}" >> cert.txt
base64 --decode cert.txt -o ${PROFILE}
rm cert.txt
# run the ruby build script
if ! [ -x "$(command -v xcodebuild)" ]; then
    echo 'Error: xcodebuild is not installed.'
    exit 1
fi
PROFILE=PyTorch_CI_2021
ruby ${PROJ_ROOT}/scripts/xcode_build.rb -i ${PROJ_ROOT}/build_ios/install -x ${PROJ_ROOT}/ios/TestApp/TestApp.xcodeproj -p ${IOS_PLATFORM} -c ${PROFILE} -t ${IOS_DEV_TEAM_ID}
