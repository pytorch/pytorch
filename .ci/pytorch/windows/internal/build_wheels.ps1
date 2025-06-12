& "windows/internal/vc_install_helper.bat"
if ($LASTEXITCODE -ne 0) { exit 1 }

& "windows/internal/cuda_install.bat"
if ($LASTEXITCODE -ne 0) { exit 1 }

& "windows/internal/xpu_install.bat"
if ($LASTEXITCODE -ne 0) { exit 1 }

. "windows/build_pytorch.ps1" $env:CUDA_VERSION $env:PYTORCH_BUILD_VERSION $env:PYTORCH_BUILD_NUMBER
