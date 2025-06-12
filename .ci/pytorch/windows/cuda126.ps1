# Import common functions
. "$PSScriptRoot\common.ps1"

# The setup.py/MODULE_NAME check, clone/clean, and check_deps are already handled in common.ps1

# Check for optional components
$env:USE_CUDA = ""
$env:CMAKE_GENERATOR = "Visual Studio 15 2017 Win64"

# Check and set NVTOOLSEXT_PATH
if (-not $env:NVTOOLSEXT_PATH) {
    $nvtoolsPath = "C:\Program Files\NVIDIA Corporation\NvToolsExt\lib\x64\nvToolsExt64_1.lib"
    if (Test-Path $nvtoolsPath) {
        $env:NVTOOLSEXT_PATH = "C:\Program Files\NVIDIA Corporation\NvToolsExt"
    } else {
        Write-Error "NVTX (Visual Studio Extension for CUDA) not installed, failing"
        exit 1
    }
}

# Check and set CUDA_PATH_V126
if (-not $env:CUDA_PATH_V126) {
    $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe"
    if (Test-Path $cudaPath) {
        $env:CUDA_PATH_V126 = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
    } else {
        Write-Error "CUDA 12.6 not found, failing"
        exit 1
    }
}

# Set CUDA compilation flags based on BUILD_VISION
if (-not $env:BUILD_VISION) {
    $env:TORCH_CUDA_ARCH_LIST = "6.1;7.0;7.5;8.0;8.6;9.0"
    $env:TORCH_NVCC_FLAGS = "-Xfatbin -compress-all"
} else {
    $env:NVCC_FLAGS = "-D__CUDA_NO_HALF_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_90,code=compute_90"
}

# Set CUDA environment
$env:CUDA_PATH = $env:CUDA_PATH_V126
$env:PATH = "$env:CUDA_PATH_V126\bin;$env:PATH"

# Check optional components
Invoke-ExternalCommand "$PSScriptRoot\internal\check_opts.bat"

# Check if NIGHTLIES_PYTORCH_ROOT exists and change directory if it does
if (Test-Path env:NIGHTLIES_PYTORCH_ROOT) {
    $nightliesParent = Split-Path $env:NIGHTLIES_PYTORCH_ROOT -Parent
    Set-Location $nightliesParent
}

Invoke-ExternalCommand "$PSScriptRoot\internal\copy.bat"
Invoke-ExternalCommand "$PSScriptRoot\internal\setup.bat"
