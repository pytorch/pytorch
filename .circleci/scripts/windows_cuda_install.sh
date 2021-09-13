#!/bin/bash
set -eux -o pipefail

cuda_major_version=${CUDA_VERSION%.*}

if [[ "$cuda_major_version" == "10" ]]; then
    cuda_installer_name="cuda_10.1.243_426.00_win10"
    cuda_install_packages="nvcc_10.1 cuobjdump_10.1 nvprune_10.1 cupti_10.1 cublas_10.1 cublas_dev_10.1 cudart_10.1 cufft_10.1 cufft_dev_10.1 curand_10.1 curand_dev_10.1 cusolver_10.1 cusolver_dev_10.1 cusparse_10.1 cusparse_dev_10.1 nvgraph_10.1 nvgraph_dev_10.1 npp_10.1 npp_dev_10.1 nvrtc_10.1 nvrtc_dev_10.1 nvml_dev_10.1"
elif [[ "$cuda_major_version" == "11" ]]; then
    cuda_installer_name="cuda_11.1.0_456.43_win10"
    cuda_install_packages="nvcc_11.1 cuobjdump_11.1 nvprune_11.1 nvprof_11.1 cupti_11.1 cublas_11.1 cublas_dev_11.1 cudart_11.1 cufft_11.1 cufft_dev_11.1 curand_11.1 curand_dev_11.1 cusolver_11.1 cusolver_dev_11.1 cusparse_11.1 cusparse_dev_11.1 npp_11.1 npp_dev_11.1 nvrtc_11.1 nvrtc_dev_11.1 nvml_dev_11.1"
else
    echo "CUDA_VERSION $CUDA_VERSION is not supported yet"
    exit 1
fi

if [[ "$cuda_major_version" == "11" && "${JOB_EXECUTOR}" == "windows-with-nvidia-gpu" ]]; then
    cuda_install_packages="${cuda_install_packages} Display.Driver"
fi


if [[ -f "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/bin/nvcc.exe" ]]; then
    echo "Existing CUDA v${CUDA_VERSION} installation found, skipping install"
else
    tmp_dir=$(mktemp -d)
    (
        # no need to popd after, the subshell shouldn't affect the parent shell
        pushd "${tmp_dir}"
        cuda_installer_link="https://ossci-windows.s3.amazonaws.com/${cuda_installer_name}.exe"

        curl --retry 3 -kLO $cuda_installer_link
        7z x ${cuda_installer_name}.exe -o${cuda_installer_name}
        pushd ${cuda_installer_name}
        mkdir cuda_install_logs

        set +e

        # This breaks for some reason if you quote cuda_install_packages
        # shellcheck disable=SC2086
        ./setup.exe -s ${cuda_install_packages} -loglevel:6 -log:"$(pwd -W)/cuda_install_logs"

        set -e

        if [[ ! -f "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/bin/nvcc.exe" ]]; then
            echo "CUDA installation failed"
            mkdir -p /c/w/build-results
            7z a "c:\\w\\build-results\\cuda_install_logs.7z" cuda_install_logs
            exit 1
        fi
    )
    rm -rf "${tmp_dir}"
fi

if [[ -f "/c/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64/nvToolsExt64_1.dll" ]]; then
    echo "Existing nvtools installation found, skipping install"
else
    # create tmp dir for download
    tmp_dir=$(mktemp -d)
    (
        # no need to popd after, the subshell shouldn't affect the parent shell
        pushd "${tmp_dir}"
        curl --retry 3 -kLO https://ossci-windows.s3.amazonaws.com/NvToolsExt.7z
        7z x NvToolsExt.7z -oNvToolsExt
        mkdir -p "C:/Program Files/NVIDIA Corporation/NvToolsExt"
        cp -r NvToolsExt/* "C:/Program Files/NVIDIA Corporation/NvToolsExt/"
    )
    rm -rf "${tmp_dir}"
fi
