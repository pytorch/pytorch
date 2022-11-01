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
target_libs=(libc10.a libclog.a libcpuinfo.a libeigen_blas.a libpthreadpool.a libpytorch_qnnpack.a libtorch_cpu.a libtorch.a libXNNPACK.a)
for lib in ${target_libs[*]}
do
    if [ -f "${ARTIFACTS_DIR}/x86_64/lib/${lib}" ] && [ -f "${ARTIFACTS_DIR}/arm64/lib/${lib}" ]; then
        libs=("${ARTIFACTS_DIR}/x86_64/lib/${lib}" "${ARTIFACTS_DIR}/arm64/lib/${lib}")
        lipo -create "${libs[@]}" -o ${ZIP_DIR}/install/lib/${lib}
    fi
done
lipo -i ${ZIP_DIR}/install/lib/*.a
echo "BUILD_LITE_INTERPRETER: ${BUILD_LITE_INTERPRETER}"
# copy the umbrella header and license
if [ "${BUILD_LITE_INTERPRETER}" == "1" ]; then
    cp ${PROJ_ROOT}/ios/LibTorch-Lite.h ${ZIP_DIR}/src/
else
    cp ${PROJ_ROOT}/ios/LibTorch.h ${ZIP_DIR}/src/
fi
cp ${PROJ_ROOT}/LICENSE ${ZIP_DIR}/
# zip the library
#export DATE="$(date -u +%Y%m%d)"
#export IOS_NIGHTLY_BUILD_VERSION="1.13.0.${DATE}"
if [ "${BUILD_LITE_INTERPRETER}" == "1" ]; then
    # libtorch_lite_ios_nightly_1.11.0.20210810.zip
    #ZIPFILE="libtorch_lite_ios_nightly_${IOS_NIGHTLY_BUILD_VERSION}.zip"
    ZIPFILE="libtorch_lite_ios_1.13.0.zip"
else
    #ZIPFILE="libtorch_ios_nightly_build.zip"
    ZIPFILE="libtorch_ios_1.13.0.zip"
fi
cd ${ZIP_DIR}
#for testing
touch version.txt
echo "1.13.0" > version.txt
zip -r ${ZIPFILE} install src version.txt LICENSE
# upload to aws
# Install conda then 'conda install' awscli
curl --retry 3 -o ~/conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
chmod +x ~/conda.sh
/bin/bash ~/conda.sh -b -p ~/anaconda
export PATH="~/anaconda/bin:${PATH}"
source ~/anaconda/bin/activate
conda install -c conda-forge awscli --yes
set +x
export AWS_ACCESS_KEY_ID=${AWS_S3_ACCESS_KEY_FOR_PYTORCH_BINARY_UPLOAD}
export AWS_SECRET_ACCESS_KEY=${AWS_S3_ACCESS_SECRET_FOR_PYTORCH_BINARY_UPLOAD}
set +x
# echo "AWS KEY: ${AWS_ACCESS_KEY_ID}"
# echo "AWS SECRET: ${AWS_SECRET_ACCESS_KEY}"
aws s3 cp ${ZIPFILE} s3://ossci-ios/ --acl public-read
