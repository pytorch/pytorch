@echo on

if "%CUDA_VERSION%" == "cpu" (
    echo Skipping for CPU builds
    exit /b 0
)

set SRC_DIR=%~dp0\..

if not exist "%SRC_DIR%\temp_build" mkdir "%SRC_DIR%\temp_build"

set /a CUDA_VER=%CUDA_VERSION%

if %CUDA_VER% EQU 9 goto cuda92
if %CUDA_VER% EQU 10 goto cuda101

echo CUDA %CUDA_VER% is not supported
exit /b 1

:cuda92
set CUDA_VER_MAJOR=9
set CUDA_VER_MINOR=2
set CUDA_VERSION_STR=9.2
if not exist "%SRC_DIR%\temp_build\cuda_9.2.148_win10.exe" (
    curl -k -L https://ossci-windows.s3.amazonaws.com/win2016/cuda_9.2.148_win10.exe --output "%SRC_DIR%\temp_build\cuda_9.2.148_win10.exe"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\cuda_9.2.148_win10.exe"
    set "ARGS=nvcc_9.2 cuobjdump_9.2 nvprune_9.2 cupti_9.2 cublas_9.2 cublas_dev_9.2 cudart_9.2 cufft_9.2 cufft_dev_9.2 curand_9.2 curand_dev_9.2 cusolver_9.2 cusolver_dev_9.2 cusparse_9.2 cusparse_dev_9.2 nvgraph_9.2 nvgraph_dev_9.2 npp_9.2 npp_dev_9.2 nvrtc_9.2 nvrtc_dev_9.2 nvml_dev_9.2"
)

if not exist "%SRC_DIR%\temp_build\cudnn-9.2-windows10-x64-v7.2.1.38.zip" (
    curl -k -L https://ossci-windows.s3.amazonaws.com/win2016/cudnn-9.2-windows10-x64-v7.2.1.38.zip --output "%SRC_DIR%\temp_build\cudnn-9.2-windows10-x64-v7.2.1.38.zip"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\cudnn-9.2-windows10-x64-v7.2.1.38.zip"
)
set NVCC_FLAGS=-D__CUDA_NO_HALF_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_50,code=compute_50

goto cuda_common

:cuda100

set CUDA_VER_MAJOR=10
set CUDA_VER_MINOR=0
set CUDA_VERSION_STR=10.0
if not exist "%SRC_DIR%\temp_build\cuda_10.0.130_411.31_win10.exe" (
    curl -k -L https://ossci-windows.s3.amazonaws.com/win2016/cuda_10.0.130_411.31_win10.exe --output "%SRC_DIR%\temp_build\cuda_10.0.130_411.31_win10.exe"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\cuda_10.0.130_411.31_win10.exe"
    set "ARGS=nvcc_10.0 cuobjdump_10.0 nvprune_10.0 cupti_10.0 cublas_10.0 cublas_dev_10.0 cudart_10.0 cufft_10.0 cufft_dev_10.0 curand_10.0 curand_dev_10.0 cusolver_10.0 cusolver_dev_10.0 cusparse_10.0 cusparse_dev_10.0 nvgraph_10.0 nvgraph_dev_10.0 npp_10.0 npp_dev_10.0 nvrtc_10.0 nvrtc_dev_10.0 nvml_dev_10.0"
)

if not exist "%SRC_DIR%\temp_build\cudnn-10.0-windows10-x64-v7.4.1.5.zip" (
    curl -k -L https://ossci-windows.s3.amazonaws.com/win2016/cudnn-10.0-windows10-x64-v7.4.1.5.zip --output "%SRC_DIR%\temp_build\cudnn-10.0-windows10-x64-v7.4.1.5.zip"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\cudnn-10.0-windows10-x64-v7.4.1.5.zip"
)

goto cuda_common

:cuda101

set CUDA_VER_MAJOR=10
set CUDA_VER_MINOR=1
set CUDA_VERSION_STR=10.1
if not exist "%SRC_DIR%\temp_build\cuda_10.1.243_426.00_win10.exe" (
    curl -k -L https://ossci-windows.s3.amazonaws.com/cuda_10.1.243_426.00_win10.exe --output "%SRC_DIR%\temp_build\cuda_10.1.243_426.00_win10.exe"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\cuda_10.1.243_426.00_win10.exe"
    set "ARGS=nvcc_10.1 cuobjdump_10.1 nvprune_10.1 cupti_10.1 cublas_10.1 cublas_dev_10.1 cudart_10.1 cufft_10.1 cufft_dev_10.1 curand_10.1 curand_dev_10.1 cusolver_10.1 cusolver_dev_10.1 cusparse_10.1 cusparse_dev_10.1 nvgraph_10.1 nvgraph_dev_10.1 npp_10.1 npp_dev_10.1 nvrtc_10.1 nvrtc_dev_10.1 nvml_dev_10.1"
)

if not exist "%SRC_DIR%\temp_build\cudnn-10.1-windows10-x64-v7.6.4.38.zip" (
    curl -k -L https://ossci-windows.s3.amazonaws.com/cudnn-10.1-windows10-x64-v7.6.4.38.zip --output "%SRC_DIR%\temp_build\cudnn-10.1-windows10-x64-v7.6.4.38.zip"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\cudnn-10.1-windows10-x64-v7.6.4.38.zip"
)

goto cuda_common

:cuda102

set CUDA_VER_MAJOR=10
set CUDA_VER_MINOR=2
set CUDA_VERSION_STR=10.2
if not exist "%SRC_DIR%\temp_build\cuda_10.2.89_441.22_win10.exe" (
    curl -k -L https://ossci-windows.s3.amazonaws.com/cuda_10.2.89_441.22_win10.exe --output "%SRC_DIR%\temp_build\cuda_10.2.89_441.22_win10.exe"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\cuda_10.2.89_441.22_win10.exe"
    set "ARGS=nvcc_10.2 cuobjdump_10.2 nvprune_10.2 cupti_10.2 cublas_10.2 cublas_dev_10.2 cudart_10.2 cufft_10.2 cufft_dev_10.2 curand_10.2 curand_dev_10.2 cusolver_10.2 cusolver_dev_10.2 cusparse_10.2 cusparse_dev_10.2 nvgraph_10.2 nvgraph_dev_10.2 npp_10.2 npp_dev_10.2 nvrtc_10.2 nvrtc_dev_10.2 nvml_dev_10.2"
)

if not exist "%SRC_DIR%\temp_build\cudnn-10.2-windows10-x64-v7.6.5.32.zip" (
    curl -k -L https://ossci-windows.s3.amazonaws.com/cudnn-10.2-windows10-x64-v7.6.5.32.zip --output "%SRC_DIR%\temp_build\cudnn-10.2-windows10-x64-v7.6.5.32.zip"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\cudnn-10.2-windows10-x64-v7.6.5.32.zip"
)

goto cuda_common

:cuda_common

if not exist "%SRC_DIR%\temp_build\NvToolsExt.7z" (
    curl -k -L https://www.dropbox.com/s/9mcolalfdj4n979/NvToolsExt.7z?dl=1 --output "%SRC_DIR%\temp_build\NvToolsExt.7z"
    if errorlevel 1 exit /b 1
)

if not exist "%SRC_DIR%\temp_build\gpu_driver_dlls.7z" (
    curl -k -L "https://drive.google.com/u/0/uc?id=1injUyo3lnarMgWyRcXqKg4UGnN0ysmuq&export=download" --output "%SRC_DIR%\temp_build\gpu_driver_dlls.zip"
    if errorlevel 1 exit /b 1
)

echo Installing CUDA toolkit...
7z x %CUDA_SETUP_FILE% -o"%SRC_DIR%\temp_build\cuda"
pushd "%SRC_DIR%\temp_build\cuda"
start /wait setup.exe -s %ARGS%
popd

echo Installing VS integration...
if "%VC_YEAR%" == "2017" (
    xcopy /Y "%SRC_DIR%\temp_build\cuda\CUDAVisualStudioIntegration\extras\visual_studio_integration\MSBuildExtensions\*.*" "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\Common7\IDE\VC\VCTargets\BuildCustomizations"
)
if "%VC_YEAR%" == "2019" (
    xcopy /Y "%SRC_DIR%\temp_build\cuda\CUDAVisualStudioIntegration\extras\visual_studio_integration\MSBuildExtensions\*.*" "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Microsoft\VC\v160\BuildCustomizations"
)

echo Installing NvToolsExt...
7z x %SRC_DIR%\temp_build\NvToolsExt.7z -o"%SRC_DIR%\temp_build\NvToolsExt"
mkdir "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\bin\x64"
mkdir "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\include"
mkdir "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\lib\x64"
xcopy /Y "%SRC_DIR%\temp_build\NvToolsExt\bin\x64\*.*" "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\bin\x64"
xcopy /Y "%SRC_DIR%\temp_build\NvToolsExt\include\*.*" "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\include"
xcopy /Y "%SRC_DIR%\temp_build\NvToolsExt\lib\x64\*.*" "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\lib\x64"

echo Setting up environment...
set "PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\bin;%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\libnvvp;%PATH%"
set "CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%"
set "CUDA_PATH_V%CUDA_VER_MAJOR%_%CUDA_VER_MINOR%=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%"
set "NVTOOLSEXT_PATH=%ProgramFiles%\NVIDIA Corporation\NvToolsExt"

if not exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\bin\nvcc.exe" (
    echo CUDA %CUDA_VERSION_STR% installed failed.
    exit /b 1
)

echo Installing cuDNN...
7z x %CUDNN_SETUP_FILE% -o"%SRC_DIR%\temp_build\cudnn"
xcopy /Y "%SRC_DIR%\temp_build\cudnn\cuda\bin\*.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\bin"
xcopy /Y "%SRC_DIR%\temp_build\cudnn\cuda\lib\x64\*.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\lib\x64"
xcopy /Y "%SRC_DIR%\temp_build\cudnn\cuda\include\*.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\include"

echo Installing GPU driver DLLs
7z x %SRC_DIR%\temp_build\gpu_driver_dlls.zip -o"C:\Windows\System32"

echo Cleaning temp files
rd /s /q "%SRC_DIR%\temp_build" || ver > nul
