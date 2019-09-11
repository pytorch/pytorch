echo "PWD: `pwd`"
WORKSPACE=/Users/distiller/workspace
PROJ_ROOT=/Users/distiller/project
ARTIFACTS_DIR=${WORKSPACE}/ios
ls ${ARTIFACTS_DIR}
ZIP_DIR=${WORKSPACE}/zip
mkdir -p ${ZIP_DIR}/install/lib
mkdir -p ${ZIP_DIR}/src
# copy header files
cp -R ${ARTIFACTS_DIR}/arm64/include ${ZIP_DIR}/install/
# archvie to FAT bianry
libs=(${ARTIFACTS_DIR}/x86_64/lib/libtorch_x86_64.a ${ARTIFACTS_DIR}/arm64/lib/libtorch_arm64.a ${ARTIFACTS_DIR}/armv7s/lib/libtorch_armv7s.a )
lipo -create ${libs[@]} -o ${ZIP_DIR}/install/lib/libtorch.a
lipo -i ${ZIP_DIR}/install/lib/libtorch.a
# copy the umbrella header
if [ -e ${PROJ_ROOT}/ios/LibTorch.h ]; then
    cp ${PROJ_ROOT}/ios/LibTorch.h ${ZIP_DIR}/src/LibTorch.h
else
    echo "LibTorch.h not found!"
    touch ${ZIP_DIR}/src/LibTorch.h
    echo "import <torch/script.h>" > ${ZIP_DIR}/src/LibTorch.h
fi
# zip the library
export ZIPFILE=libtorch_ios_nightly_build.zip
cd ${ZIP_DIR}
zip -r ${ZIPFILE} install src
# upload to aws
brew install awscli
set +x
export AWS_ACCESS_KEY_ID=${AWS_S3_ACCESS_KEY_FOR_PYTORCH_BINARY_UPLOAD}
export AWS_SECRET_ACCESS_KEY=${AWS_S3_ACCESS_SECRET_FOR_PYTORCH_BINARY_UPLOAD}
set +x
# echo "AWS KEY: ${AWS_ACCESS_KEY_ID}"
# echo "AWS SECRET: ${AWS_SECRET_ACCESS_KEY}"  
aws s3 cp ${ZIPFILE} s3://ossci-ios-build/ --acl public-read