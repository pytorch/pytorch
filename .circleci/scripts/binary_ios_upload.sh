echo ""
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
cd ${ZIP_DIR}/install/lib
target_libs=(libc10.a libclog.a libcpuinfo.a libqnnpack.a libtorch.a)
for lib in ${target_libs[*]}
do
    libs=(${ARTIFACTS_DIR}/x86_64/lib/${lib} ${ARTIFACTS_DIR}/arm64/lib/${lib} ${ARTIFACTS_DIR}/armv7s/lib/${lib} )
    lipo -create ${libs[@]} -o ${ZIP_DIR}/install/lib/${lib}
done
# for nnpack, we only support arm64/armv7s build
lipo -create ${ARTIFACTS_DIR}/arm64/lib/libnnpack.a ${ARTIFACTS_DIR}/armv7s/lib/libnnpack.a -o ${ZIP_DIR}/install/lib/libnnpack.a
lipo -i ${ZIP_DIR}/install/lib/*.a
# copy the umbrella header
if [ -e ${PROJ_ROOT}/ios/LibTorch.h ]; then
    cp ${PROJ_ROOT}/ios/LibTorch.h ${ZIP_DIR}/src/
    cp ${PROJ_ROOT}/ios/README.md ${ZIP_DIR}/src/
else
    echo "LibTorch.h not found!"
    touch ${ZIP_DIR}/src/LibTorch.h
    echo "import <torch/script.h>" > ${ZIP_DIR}/src/LibTorch.h
fi
# zip the library
timestamp=`date +"%Y-%m-%d_%H-%M-%S"`
ZIPFILE=libtorch_ios_nightly_build.zip
cd ${ZIP_DIR}
touch version.txt
echo `date +%s` > version.txt
zip -r ${ZIPFILE} install src version.txt
# upload to aws
brew install awscli
set +x
export AWS_ACCESS_KEY_ID=${AWS_S3_ACCESS_KEY_FOR_PYTORCH_BINARY_UPLOAD}
export AWS_SECRET_ACCESS_KEY=${AWS_S3_ACCESS_SECRET_FOR_PYTORCH_BINARY_UPLOAD}
set +x
# echo "AWS KEY: ${AWS_ACCESS_KEY_ID}"
# echo "AWS SECRET: ${AWS_SECRET_ACCESS_KEY}"  
aws s3 cp ${ZIPFILE} s3://ossci-ios-build/ --acl public-read