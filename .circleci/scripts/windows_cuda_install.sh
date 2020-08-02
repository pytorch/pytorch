#!/bin/bash
set -eux -o pipefail

curl --retry 3 -kLO https://ossci-windows.s3.amazonaws.com/cuda_11.0.2_451.48_win10.exe
7z x cuda_11.0.2_451.48_win10.exe -ocuda_11.0.2_451.48_win10
cd cuda_11.0.2_451.48_win10
mkdir cuda_install_logs

set +e

./setup.exe -s nvcc_11.0 cuobjdump_11.0 nvprune_11.0 cupti_11.0 cublas_11.0 cublas_dev_11.0 cudart_11.0 cufft_11.0 cufft_dev_11.0 curand_11.0 curand_dev_11.0 cusolver_11.0 cusolver_dev_11.0 cusparse_11.0 cusparse_dev_11.0 npp_11.0 npp_dev_11.0 nvrtc_11.0 nvrtc_dev_11.0 nvml_dev_11.0 -loglevel:6 -log:"$(pwd -W)/cuda_install_logs"

set -e

if [[ "${VC_YEAR}" == "2017" ]]; then
    cp -r visual_studio_integration/CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions/* "C:/Program Files (x86)/Microsoft Visual Studio/2017/${VC_PRODUCT}/Common7/IDE/VC/VCTargets/BuildCustomizations/"
else
    cp -r visual_studio_integration/CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions/* "C:/Program Files (x86)/Microsoft Visual Studio/2019/${VC_PRODUCT}/MSBuild/Microsoft/VC/v160/BuildCustomizations/"
fi

if ! ls "/c/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64/nvToolsExt64_1.dll"
then
    curl --retry 3 -kLO https://ossci-windows.s3.amazonaws.com/NvToolsExt.7z
    7z x NvToolsExt.7z -oNvToolsExt
    mkdir -p "C:/Program Files/NVIDIA Corporation/NvToolsExt"
    cp -r NvToolsExt/* "C:/Program Files/NVIDIA Corporation/NvToolsExt/"
    export NVTOOLSEXT_PATH="C:\\Program Files\\NVIDIA Corporation\\NvToolsExt\\"
fi

if ! ls "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin/nvcc.exe"
then
    echo "CUDA installation failed"
    mkdir -p /c/w/build-results
    7z a "c:\\w\\build-results\\cuda_install_logs.7z" cuda_install_logs
    exit 1
fi

cd ..
rm -rf ./cuda_11.0.2_451.48_win10
rm -f ./cuda_11.0.2_451.48_win10.exe
