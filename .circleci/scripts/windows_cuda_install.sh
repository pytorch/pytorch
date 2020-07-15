#!/bin/bash
set -eux -o pipefail

curl --retry 3 -kLO https://ossci-windows.s3.amazonaws.com/cuda_10.1.243_426.00_win10.exe
7z x cuda_10.1.243_426.00_win10.exe -ocuda_10.1.243_426.00_win10
cd cuda_10.1.243_426.00_win10
mkdir cuda_install_logs

set +e

./setup.exe -s nvcc_10.1 cuobjdump_10.1 nvprune_10.1 cupti_10.1 cublas_10.1 cublas_dev_10.1 cudart_10.1 cufft_10.1 cufft_dev_10.1 curand_10.1 curand_dev_10.1 cusolver_10.1 cusolver_dev_10.1 cusparse_10.1 cusparse_dev_10.1 nvgraph_10.1 nvgraph_dev_10.1 npp_10.1 npp_dev_10.1 nvrtc_10.1 nvrtc_dev_10.1 nvml_dev_10.1 -loglevel:6 -log:"$(pwd -W)/cuda_install_logs"

set -e

if [[ "${VC_YEAR}" == "2017" ]]; then
    cp -r CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions/* "C:/Program Files (x86)/Microsoft Visual Studio/2017/${VC_PRODUCT}/Common7/IDE/VC/VCTargets/BuildCustomizations/"
else
    cp -r CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions/* "C:/Program Files (x86)/Microsoft Visual Studio/2019/${VC_PRODUCT}/MSBuild/Microsoft/VC/v160/BuildCustomizations/"
fi

curl --retry 3 -kLO https://ossci-windows.s3.amazonaws.com/NvToolsExt.7z
7z x NvToolsExt.7z -oNvToolsExt
mkdir -p "C:/Program Files/NVIDIA Corporation/NvToolsExt"
cp -r NvToolsExt/* "C:/Program Files/NVIDIA Corporation/NvToolsExt/"
export NVTOOLSEXT_PATH="C:\\Program Files\\NVIDIA Corporation\\NvToolsExt\\"

if ! ls "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin/nvcc.exe"
then
    echo "CUDA installation failed"
    mkdir -p /c/w/build-results
    7z a "c:\\w\\build-results\\cuda_install_logs.7z" cuda_install_logs
    exit 1
fi

cd ..
rm -rf ./cuda_10.1.243_426.00_win10
rm -f ./cuda_10.1.243_426.00_win10.exe
