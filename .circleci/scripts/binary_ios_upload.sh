#!/bin/bash
set -ex -o pipefail

echo ""
echo "DIR: $(pwd)"
WORKSPACE=/Users/distiller/workspace
PROJ_ROOT=/Users/distiller/project
ARTIFACTS_DIR=${WORKSPACE}/ios
ls ${ARTIFACTS_DIR}
ZIP_DIR=${WORKSPACE}/zip
mkdir -p ${ZIP_DIR}/install/lib
mkdir -p ${ZIP_DIR}/src
# copy header files
cp -R ${ARTIFACTS_DIR}/arm64/include ${ZIP_DIR}/install/
# build a FAT bianry
cd ${ZIP_DIR}/install/lib
target_libs=(libc10.a libclog.a libcpuinfo.a libeigen_blas.a libpytorch_qnnpack.a libtorch.a)
for lib in ${target_libs[*]}
do
    libs=(${ARTIFACTS_DIR}/x86_64/lib/${lib} ${ARTIFACTS_DIR}/arm64/lib/${lib})
    lipo -create "${libs[@]}" -o ${ZIP_DIR}/install/lib/${lib}
done
# for nnpack, we only support arm64 build
cp ${ARTIFACTS_DIR}/arm64/lib/libnnpack.a ./
lipo -i ${ZIP_DIR}/install/lib/*.a
# copy the umbrella header and license
cp ${PROJ_ROOT}/ios/LibTorch.h ${ZIP_DIR}/src/
cp ${PROJ_ROOT}/LICENSE ${ZIP_DIR}/
# zip the library
# ZIPFILE=libtorch_ios_nightly_build.zip
ZIPFILE=libtorch_ios_1.3.1.zip
cd ${ZIP_DIR}
#for testing
touch version.txt
echo $(date +%s) > version.txt
zip -r ${ZIPFILE} install src version.txt LICENSE
# upload to aws
brew install awscli
set +x
export AWS_ACCESS_KEY_ID=${AWS_S3_ACCESS_KEY_FOR_PYTORCH_BINARY_UPLOAD}
export AWS_SECRET_ACCESS_KEY=${AWS_S3_ACCESS_SECRET_FOR_PYTORCH_BINARY_UPLOAD}
set +x
# echo "AWS KEY: ${AWS_ACCESS_KEY_ID}"
# echo "AWS SECRET: ${AWS_SECRET_ACCESS_KEY}"
# aws s3 cp ${ZIPFILE} s3://ossci-ios-build/ --acl public-read
aws s3 cp ${ZIPFILE} s3://ossci-ios/ --acl public-read
