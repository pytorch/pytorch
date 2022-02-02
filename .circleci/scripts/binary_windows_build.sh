#!/bin/bash
set -eux -o pipefail

source "${BINARY_ENV_FILE:-/c/w/env}"
mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR"

export CUDA_VERSION="${DESIRED_CUDA/cu/}"
export USE_SCCACHE=1
export SCCACHE_BUCKET=ossci-compiler-cache-windows
export NIGHTLIES_PYTORCH_ROOT="$PYTORCH_ROOT"
export VC_YEAR=2019

if [[ "${DESIRED_CUDA}" == *"cu11"* ]]; then
    export BUILD_SPLIT_CUDA=ON
fi

echo "Free Space for CUDA DEBUG BUILD"
if [[ "${CIRCLECI:-}" == 'true' ]]; then
    if [[ -d "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community" ]]; then
        rm -rf "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community"
    fi

    if [[ -d "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0" ]]; then
        rm -rf "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0"
    fi

    if [[ -d "C:\\Program Files (x86)\\Microsoft.NET" ]]; then
        rm -rf "C:\\Program Files (x86)\\Microsoft.NET"
    fi

    if [[ -d "C:\\Program Files\\dotnet" ]]; then
        rm -rf "C:\\Program Files\\dotnet"
    fi

    if [[ -d "C:\\Program Files (x86)\\dotnet" ]]; then
        rm -rf "C:\\Program Files (x86)\\dotnet"
    fi

    if [[ -d "C:\\Program Files (x86)\\Microsoft SQL Server" ]]; then
        rm -rf "C:\\Program Files (x86)\\Microsoft SQL Server"
    fi

    if [[ -d "C:\\Program Files (x86)\\Xamarin" ]]; then
        rm -rf "C:\\Program Files (x86)\\Xamarin"
    fi

    if [[ -d "C:\\Program Files (x86)\\Google" ]]; then
        rm -rf "C:\\Program Files (x86)\\Google"
    fi
    set +x
    export AWS_ACCESS_KEY_ID=${CIRCLECI_AWS_ACCESS_KEY_FOR_SCCACHE_S3_BUCKET_V4:-}
    export AWS_SECRET_ACCESS_KEY=${CIRCLECI_AWS_SECRET_KEY_FOR_SCCACHE_S3_BUCKET_V4:-}
    set -x
    if [[ -d "C:\\ProgramData\\Microsoft\\VisualStudio\\Packages\\_Instances" ]]; then
        mv "C:\\ProgramData\\Microsoft\\VisualStudio\\Packages\\_Instances" .
        rm -rf "C:\\ProgramData\\Microsoft\\VisualStudio\\Packages"
        mkdir -p "C:\\ProgramData\\Microsoft\\VisualStudio\\Packages"
        mv _Instances "C:\\ProgramData\\Microsoft\\VisualStudio\\Packages"
    fi
    if [[ -d "C:\\Microsoft" ]]; then
        # don't use quotes here
        rm -rf /c/Microsoft/AndroidNDK*
    fi
fi

echo "Free space on filesystem before build:"
df -h

pushd "$BUILDER_ROOT"
if [[ "$PACKAGE_TYPE" == 'conda' ]]; then
    ./windows/internal/build_conda.bat
elif [[ "$PACKAGE_TYPE" == 'wheel' || "$PACKAGE_TYPE" == 'libtorch' ]]; then
    ./windows/internal/build_wheels.bat
fi

echo "Free space on filesystem after build:"
df -h
