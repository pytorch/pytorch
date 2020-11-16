#!/bin/bash
set -eux -o pipefail

if [[ "$CUDA_VERSION" == "10" ]]; then
    cuda_complete_version="10.1"
    cuda_installer_name="cuda_10.1.243_426.00_win10"
    msbuild_project_dir="CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions"
    cuda_install_packages="nvcc_10.1 cuobjdump_10.1 nvprune_10.1 cupti_10.1 cublas_10.1 cublas_dev_10.1 cudart_10.1 cufft_10.1 cufft_dev_10.1 curand_10.1 curand_dev_10.1 cusolver_10.1 cusolver_dev_10.1 cusparse_10.1 cusparse_dev_10.1 nvgraph_10.1 nvgraph_dev_10.1 npp_10.1 npp_dev_10.1 nvrtc_10.1 nvrtc_dev_10.1 nvml_dev_10.1"
elif [[ "$CUDA_VERSION" == "11" ]]; then
    cuda_complete_version="11.0"
    cuda_installer_name="cuda_11.0.2_451.48_win10"
    msbuild_project_dir="visual_studio_integration/CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions"
    cuda_install_packages="nvcc_11.0 cuobjdump_11.0 nvprune_11.0 nvprof_11.0 cupti_11.0 cublas_11.0 cublas_dev_11.0 cudart_11.0 cufft_11.0 cufft_dev_11.0 curand_11.0 curand_dev_11.0 cusolver_11.0 cusolver_dev_11.0 cusparse_11.0 cusparse_dev_11.0 npp_11.0 npp_dev_11.0 nvrtc_11.0 nvrtc_dev_11.0 nvml_dev_11.0"
else
    echo "CUDA_VERSION $CUDA_VERSION is not supported yet"
    exit 1
fi

cuda_installer_link="https://ossci-windows.s3.amazonaws.com/${cuda_installer_name}.exe"

curl --retry 3 -kLO $cuda_installer_link
7z x ${cuda_installer_name}.exe -o${cuda_installer_name}
cd ${cuda_installer_name}
mkdir cuda_install_logs

set +e

./setup.exe -s ${cuda_install_packages} -loglevel:6 -log:"$(pwd -W)/cuda_install_logs"

set -e

if [[ "${VC_YEAR}" == "2017" ]]; then
    cp -r ${msbuild_project_dir}/* "C:/Program Files (x86)/Microsoft Visual Studio/2017/${VC_PRODUCT}/Common7/IDE/VC/VCTargets/BuildCustomizations/"
else
    cp -r ${msbuild_project_dir}/* "C:/Program Files (x86)/Microsoft Visual Studio/2019/${VC_PRODUCT}/MSBuild/Microsoft/VC/v160/BuildCustomizations/"
fi

if ! ls "/c/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64/nvToolsExt64_1.dll"
then
    curl --retry 3 -kLO https://ossci-windows.s3.amazonaws.com/NvToolsExt.7z
    7z x NvToolsExt.7z -oNvToolsExt
    mkdir -p "C:/Program Files/NVIDIA Corporation/NvToolsExt"
    cp -r NvToolsExt/* "C:/Program Files/NVIDIA Corporation/NvToolsExt/"
    export NVTOOLSEXT_PATH="C:\\Program Files\\NVIDIA Corporation\\NvToolsExt\\"
fi

if ! ls "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${cuda_complete_version}/bin/nvcc.exe"
then
    echo "CUDA installation failed"
    mkdir -p /c/w/build-results
    7z a "c:\\w\\build-results\\cuda_install_logs.7z" cuda_install_logs
    exit 1
fi

cd ..
rm -rf ./${cuda_installer_name}
rm -f ./${cuda_installer_name}.exe
