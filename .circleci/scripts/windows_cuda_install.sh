#!/bin/bash
set -eux -o pipefail

cuda_major_version=${CUDA_VERSION%.*}

case ${CUDA_VERSION} in
    10.1)
        cuda_installer_name="cuda_10.1.243_426.00_win10"
        msbuild_project_dir="CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions"
        cuda_install_packages="nvcc_10.1 cuobjdump_10.1 nvprune_10.1 cupti_10.1 cublas_10.1 cublas_dev_10.1 cudart_10.1 cufft_10.1 cufft_dev_10.1 curand_10.1 curand_dev_10.1 cusolver_10.1 cusolver_dev_10.1 cusparse_10.1 cusparse_dev_10.1 nvgraph_10.1 nvgraph_dev_10.1 npp_10.1 npp_dev_10.1 nvrtc_10.1 nvrtc_dev_10.1 nvml_dev_10.1"
        ;;
    10.2)
        cuda_installer_name="cuda_10.2.89_441.22_win10"
        msbuild_project_dir="CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions"
        cuda_install_packages="nvcc_10.2 cuobjdump_10.2 nvprune_10.2 cupti_10.2 cublas_10.2 cublas_dev_10.2 cudart_10.2 cufft_10.2 cufft_dev_10.2 curand_10.2 curand_dev_10.2 cusolver_10.2 cusolver_dev_10.2 cusparse_10.2 cusparse_dev_10.2 nvgraph_10.2 nvgraph_dev_10.2 npp_10.2 npp_dev_10.2 nvrtc_10.2 nvrtc_dev_10.2 nvml_dev_10.2"
        ;;
    11.1)
        cuda_installer_name="cuda_11.1.0_456.43_win10"
        msbuild_project_dir="visual_studio_integration/CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions"
        cuda_install_packages="nvcc_11.1 cuobjdump_11.1 nvprune_11.1 nvprof_11.1 cupti_11.1 cublas_11.1 cublas_dev_11.1 cudart_11.1 cufft_11.1 cufft_dev_11.1 curand_11.1 curand_dev_11.1 cusolver_11.1 cusolver_dev_11.1 cusparse_11.1 cusparse_dev_11.1 npp_11.1 npp_dev_11.1 nvrtc_11.1 nvrtc_dev_11.1 nvml_dev_11.1"
        ;;
    11.3)
        cuda_installer_name="cuda_11.3.0_465.89_win10"
        msbuild_project_dir="visual_studio_integration/CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions"
        cuda_install_packages="thrust_11.3 nvcc_11.3 cuobjdump_11.3 nvprune_11.3 nvprof_11.3 cupti_11.3 cublas_11.3 cublas_dev_11.3 cudart_11.3 cufft_11.3 cufft_dev_11.3 curand_11.3 curand_dev_11.3 cusolver_11.3 cusolver_dev_11.3 cusparse_11.3 cusparse_dev_11.3 npp_11.3 npp_dev_11.3 nvrtc_11.3 nvrtc_dev_11.3 nvml_dev_11.3"
        ;;
    *)
        echo "CUDA_VERSION $CUDA_VERSION is not supported yet"
        exit 1
        ;;
esac

if [[ -f "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/bin/nvcc.exe" ]]; then
    echo "Existing CUDA Toolkit installation found, skipping install..."
else
    cuda_installer_link="https://ossci-windows.s3.amazonaws.com/${cuda_installer_name}.exe"

    curl --retry 3 -kLO $cuda_installer_link
    7z x ${cuda_installer_name}.exe -o${cuda_installer_name}
    cd ${cuda_installer_name}
    mkdir cuda_install_logs

    (
        # subshell for +e
        set +e
        ./setup.exe -s ${cuda_install_packages} -loglevel:6 -log:"$(pwd -W)/cuda_install_logs"
    )

    if [[ ! -f "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/bin/nvcc.exe" ]]; then
        echo "CUDA installation failed"
        mkdir -p /c/w/build-results
        7z a "c:\\w\\build-results\\cuda_install_logs.7z" cuda_install_logs
        exit 1
    fi

    cd ..
    rm -rf ./${cuda_installer_name}
    rm -f ./${cuda_installer_name}.exe
fi

if [[ -f "/c/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64/nvToolsExt64_1.dll" ]]; then
    echo "Existing nvtools installation found, skipping install..."
else
    curl --retry 3 -kLO https://ossci-windows.s3.amazonaws.com/NvToolsExt.7z
    7z x NvToolsExt.7z -oNvToolsExt
    mkdir -p "C:/Program Files/NVIDIA Corporation/NvToolsExt"
    cp -r NvToolsExt/* "C:/Program Files/NVIDIA Corporation/NvToolsExt/"
    export NVTOOLSEXT_PATH="C:\\Program Files\\NVIDIA Corporation\\NvToolsExt\\"
fi

# Always copy vc integration to set env variables
cp -r ${msbuild_project_dir}/* "C:/Program Files (x86)/Microsoft Visual Studio/2019/${VC_PRODUCT}/MSBuild/Microsoft/VC/v160/BuildCustomizations/"
