#!/bin/bash
set -eux -o pipefail

cuda_major_version=${CUDA_VERSION%.*}

# cuda_installer_name
declare -a installers=(
    "10.1 cuda_10.1.243_426.00_win10"
    "11.1 cuda_11.1.0_456.43_win10"
    "11.2 cuda_11.2.2_461.33_win10"
)

for elem in "${installers[@]}"; do
    read -a strarr <<< "$elem"  # uses default whitespace IFS
    if [[ "$CUDA_VERSION" == "${strarr[0]}" ]]; then
        cuda_installer_name=${strarr[1]}
        break
    fi
done
if [ -z $cuda_installer_name ]; then
    echo "CUDA_VERSION $CUDA_VERSION is not supported yet"
    exit 1
fi

# msbuild_project_dir
declare -a msbuild_project_dir=(
    "10 CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions"
    "11 visual_studio_integration/CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions"
)

for elem in "${build_dirs[@]}"; do
    read -a strarr <<< "$elem" # uses default whitespace IFS
    if [[ "$cuda_major_version" == "${strarr[0]}" ]]; then
        msbuild_project_dir=${strarr[1]}
        break
    fi
done

# cuda_install_packages
cuda10_packages_template="nvcc_10.1 cuobjdump_10.1 nvprune_10.1 cupti_10.1 cublas_10.1 cublas_dev_10.1 cudart_10.1 cufft_10.1 cufft_dev_10.1 curand_10.1 curand_dev_10.1 cusolver_10.1 cusolver_dev_10.1 cusparse_10.1 cusparse_dev_10.1 nvgraph_10.1 nvgraph_dev_10.1 npp_10.1 npp_dev_10.1 nvrtc_10.1 nvrtc_dev_10.1 nvml_dev_10.1"

cuda11_packages_template="nvcc_11.1 cuobjdump_11.1 nvprune_11.1 nvprof_11.1 cupti_11.1 cublas_11.1 cublas_dev_11.1 cudart_11.1 cufft_11.1 cufft_dev_11.1 curand_11.1 curand_dev_11.1 cusolver_11.1 cusolver_dev_11.1 cusparse_11.1 cusparse_dev_11.1 npp_11.1 npp_dev_11.1 nvrtc_11.1 nvrtc_dev_11.1 nvml_dev_11.1"

declare -a install_packages=(
    "10, ${cuda10_packages_template}"
    "11, ${cuda11_packages_template}"
)
for elem in "${install_packages[@]}"; do
    IFS="," read -a strarr <<< "$elem" # use comma as delimiter because packages includes whitespace     
    if [[ "$cuda_major_version" == "${strarr[0]}" ]]; then
        packages_template="${strarr[1]}"
        cuda_install_packages=${packages_template//[1-9][0-9*]\.[0-9]/$CUDA_VERSION}
        break
    fi
done


if [[ "$cuda_major_version" == "11" && "${JOB_EXECUTOR}" == "windows-with-nvidia-gpu" ]]; then
    cuda_install_packages="${cuda_install_packages} Display.Driver"
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

if ! ls "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/bin/nvcc.exe"
then
    echo "CUDA installation failed"
    mkdir -p /c/w/build-results
    7z a "c:\\w\\build-results\\cuda_install_logs.7z" cuda_install_logs
    exit 1
fi

cd ..
rm -rf ./${cuda_installer_name}
rm -f ./${cuda_installer_name}.exe
