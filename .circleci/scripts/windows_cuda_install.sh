#!/bin/bash
set -eux -o pipefail

source "$(dirname "${BASH_SOURCE[0]}")/maps.sh"

cuda_major_version=${CUDA_VERSION%.*}

declare -a installers=(
    "10.1:cuda_10.1.243_426.00_win10"
    "11.1:cuda_11.1.0_456.43_win10"
    "11.2:cuda_11.2.2_461.33_win10"
    "11.3:cuda_11.3.0_465.89_win10"
)

declare -a build_dirs=(
    "10:CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions"
    "11:visual_studio_integration/CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions"
)

# https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#install-cuda-software
cuda10_packages_template="nvcc cuobjdump nvprune cupti cublas cublas_dev cudart cufft cufft_dev curand curand_dev cusolver cusolver_dev cusparse cusparse_dev nvgraph nvgraph_dev npp npp_dev nvrtc nvrtc_dev nvml_dev"

cuda11_packages_template="nvcc cuobjdump nvprune nvprof cupti cublas cublas_dev cudart cufft cufft_dev curand curand_dev cusolver cusolver_dev cusparse cusparse_dev npp npp_dev nvrtc nvrtc_dev nvml_dev"

declare -a install_packages=(
    "10.1:${cuda10_packages_template}"
    "11.1:${cuda11_packages_template}"
    "11.2:${cuda11_packages_template}"
    "11.3:${cuda11_packages_template} thrust"
)

# cuda_installer_name
map_get_value $CUDA_VERSION "${installers[@]}"
cuda_installer_name=$map_return_value

if [ -z $cuda_installer_name ]; then
    echo "CUDA_VERSION $CUDA_VERSION is not supported yet"
    exit 1
fi

# msbuild_project_dir
map_get_value $cuda_major_version "${build_dirs[@]}"
msbuild_project_dir=$map_return_value

# cuda_install_packages
map_get_value $CUDA_VERSION "${install_packages[@]}"
packages_template=$map_return_value
read -ra package_array <<< "$packages_template"
package_array=("${package_array[@]/%/_$CUDA_VERSION}") # add version suffix for each package
cuda_install_packages="${package_array[*]}"

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
