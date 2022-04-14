@echo on

if "%CU_VERSION%" == "cpu" (
    echo Skipping for CPU builds
    exit /b 0
)

set SRC_DIR=%~dp0\..

if not exist "%SRC_DIR%\temp_build" mkdir "%SRC_DIR%\temp_build"

rem in unit test workflow, we get CUDA_VERSION, for example 11.1
if defined CUDA_VERSION (
    set CUDA_VER=%CUDA_VERSION:.=%
) else (
    set CUDA_VER=%CU_VERSION:cu=%
)

set /a CUDA_VER=%CU_VERSION:cu=%
set CUDA_VER_MAJOR=%CUDA_VER:~0,-1%
set CUDA_VER_MINOR=%CUDA_VER:~-1,1%
set CUDA_VERSION_STR=%CUDA_VER_MAJOR%.%CUDA_VER_MINOR%


if %CUDA_VER% EQU 92 goto cuda92
if %CUDA_VER% EQU 100 goto cuda100
if %CUDA_VER% EQU 101 goto cuda101
if %CUDA_VER% EQU 102 goto cuda102
if %CUDA_VER% EQU 110 goto cuda110
if %CUDA_VER% EQU 111 goto cuda111
if %CUDA_VER% EQU 112 goto cuda112
if %CUDA_VER% EQU 113 goto cuda113
if %CUDA_VER% EQU 115 goto cuda115


echo CUDA %CUDA_VERSION_STR% is not supported
exit /b 1

:cuda92
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

goto cuda_common

:cuda100

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

if not exist "%SRC_DIR%\temp_build\cuda_10.1.243_426.00_win10.exe" (
    curl -k -L https://ossci-windows.s3.amazonaws.com/cuda_10.1.243_426.00_win10.exe --output "%SRC_DIR%\temp_build\cuda_10.1.243_426.00_win10.exe"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\cuda_10.1.243_426.00_win10.exe"
    set "ARGS=nvcc_10.1 cuobjdump_10.1 nvprune_10.1 cupti_10.1 cublas_10.1 cublas_dev_10.1 cudart_10.1 cufft_10.1 cufft_dev_10.1 curand_10.1 curand_dev_10.1 cusolver_10.1 cusolver_dev_10.1 cusparse_10.1 cusparse_dev_10.1 nvgraph_10.1 nvgraph_dev_10.1 npp_10.1 npp_dev_10.1 nvjpeg_10.1 nvjpeg_dev_10.1 nvrtc_10.1 nvrtc_dev_10.1 nvml_dev_10.1"
)

if not exist "%SRC_DIR%\temp_build\cudnn-10.1-windows10-x64-v7.6.4.38.zip" (
    curl -k -L https://ossci-windows.s3.amazonaws.com/cudnn-10.1-windows10-x64-v7.6.4.38.zip --output "%SRC_DIR%\temp_build\cudnn-10.1-windows10-x64-v7.6.4.38.zip"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\cudnn-10.1-windows10-x64-v7.6.4.38.zip"
)

goto cuda_common

:cuda102

if not exist "%SRC_DIR%\temp_build\cuda_10.2.89_441.22_win10.exe" (
    curl -k -L https://ossci-windows.s3.amazonaws.com/cuda_10.2.89_441.22_win10.exe --output "%SRC_DIR%\temp_build\cuda_10.2.89_441.22_win10.exe"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\cuda_10.2.89_441.22_win10.exe"
    set "ARGS=nvcc_10.2 cuobjdump_10.2 nvprune_10.2 cupti_10.2 cublas_10.2 cublas_dev_10.2 cudart_10.2 cufft_10.2 cufft_dev_10.2 curand_10.2 curand_dev_10.2 cusolver_10.2 cusolver_dev_10.2 cusparse_10.2 cusparse_dev_10.2 nvgraph_10.2 nvgraph_dev_10.2 npp_10.2 npp_dev_10.2 nvjpeg_10.2 nvjpeg_dev_10.2 nvrtc_10.2 nvrtc_dev_10.2 nvml_dev_10.2"
)

if not exist "%SRC_DIR%\temp_build\cudnn-10.2-windows10-x64-v7.6.5.32.zip" (
    curl -k -L https://ossci-windows.s3.amazonaws.com/cudnn-10.2-windows10-x64-v7.6.5.32.zip --output "%SRC_DIR%\temp_build\cudnn-10.2-windows10-x64-v7.6.5.32.zip"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\cudnn-10.2-windows10-x64-v7.6.5.32.zip"
)

rem The below only for cu102, if it's used in other version, e.g. cu111, torch.cuda.is_availabe() would be False.
if not exist "%SRC_DIR%\temp_build\gpu_driver_dlls.7z" (
    curl -k -L "https://drive.google.com/u/0/uc?id=1injUyo3lnarMgWyRcXqKg4UGnN0ysmuq&export=download" --output "%SRC_DIR%\temp_build\gpu_driver_dlls.zip"
    if errorlevel 1 exit /b 1
)

echo Installing GPU driver DLLs
7z x %SRC_DIR%\temp_build\gpu_driver_dlls.zip -aoa -o"C:\Windows\System32"

goto cuda_common

:cuda110

if not exist "%SRC_DIR%\temp_build\cuda_11.0.2_451.48_win10.exe" (
    curl -k -L https://ossci-windows.s3.amazonaws.com/cuda_11.0.2_451.48_win10.exe --output "%SRC_DIR%\temp_build\cuda_11.0.2_451.48_win10.exe"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\cuda_11.0.2_451.48_win10.exe"
    set "ARGS=nvcc_11.0 cuobjdump_11.0 nvprune_11.0 nvprof_11.0 cupti_11.0 cublas_11.0 cublas_dev_11.0 cudart_11.0 cufft_11.0 cufft_dev_11.0 curand_11.0 curand_dev_11.0 cusolver_11.0 cusolver_dev_11.0 cusparse_11.0 cusparse_dev_11.0 npp_11.0 npp_dev_11.0 nvjpeg_11.0 nvjpeg_dev_11.0 nvrtc_11.0 nvrtc_dev_11.0 nvml_dev_11.0"
)

if not exist "%SRC_DIR%\temp_build\cudnn-11.0-windows-x64-v8.0.4.30.zip" (
    curl -k -L https://ossci-windows.s3.amazonaws.com/cudnn-11.0-windows-x64-v8.0.4.30.zip --output "%SRC_DIR%\temp_build\cudnn-11.0-windows-x64-v8.0.4.30.zip"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\cudnn-11.0-windows-x64-v8.0.4.30.zip"
)

goto cuda_common

:cuda111

if not exist "%SRC_DIR%\temp_build\cuda_11.1.1_456.81_win10.exe" (
    curl -k -L https://ossci-windows.s3.amazonaws.com/cuda_11.1.1_456.81_win10.exe --output "%SRC_DIR%\temp_build\cuda_11.1.1_456.81_win10.exe"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\cuda_11.1.1_456.81_win10.exe"
    set "ARGS=nvcc_11.1 cuobjdump_11.1 nvprune_11.1 nvprof_11.1 cupti_11.1 cublas_11.1 cublas_dev_11.1 cudart_11.1 cufft_11.1 cufft_dev_11.1 curand_11.1 curand_dev_11.1 cusolver_11.1 cusolver_dev_11.1 cusparse_11.1 cusparse_dev_11.1 npp_11.1 npp_dev_11.1 nvjpeg_11.1 nvjpeg_dev_11.1 nvrtc_11.1 nvrtc_dev_11.1 nvml_dev_11.1"
)

if not exist "%SRC_DIR%\temp_build\cudnn-11.1-windows-x64-v8.0.5.39.zip" (
    curl -k -L https://ossci-windows.s3.amazonaws.com/cudnn-11.1-windows-x64-v8.0.5.39.zip --output "%SRC_DIR%\temp_build\cudnn-11.1-windows-x64-v8.0.5.39.zip"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\cudnn-11.1-windows-x64-v8.0.5.39.zip"
)

goto cuda_common

:cuda112

if not exist "%SRC_DIR%\temp_build\cuda_11.2.0_460.89_win10.exe" (
    curl -k -L https://ossci-windows.s3.amazonaws.com/cuda_11.2.0_460.89_win10.exe --output "%SRC_DIR%\temp_build\cuda_11.2.0_460.89_win10.exe"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\cuda_11.2.0_460.89_win10.exe"
    set "ARGS=nvcc_11.2 cuobjdump_11.2 nvprune_11.2 nvprof_11.2 cupti_11.2 cublas_11.2 cublas_dev_11.2 cudart_11.2 cufft_11.2 cufft_dev_11.2 curand_11.2 curand_dev_11.2 cusolver_11.2 cusolver_dev_11.2 cusparse_11.2 cusparse_dev_11.2 npp_11.2 npp_dev_11.2 nvjpeg_11.2 nvjpeg_dev_11.2 nvrtc_11.2 nvrtc_dev_11.2 nvml_dev_11.2"
)

if not exist "%SRC_DIR%\temp_build\cudnn-11.2-windows-x64-v8.1.0.77.zip" (
    curl -k -L http://s3.amazonaws.com/ossci-windows/cudnn-11.2-windows-x64-v8.1.0.77.zip --output "%SRC_DIR%\temp_build\cudnn-11.2-windows-x64-v8.1.0.77.zip"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\cudnn-11.2-windows-x64-v8.1.0.77.zip"
)

goto cuda_common

:cuda113

set CUDA_INSTALL_EXE=cuda_11.3.0_465.89_win10.exe
if not exist "%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%" (
    curl -k -L "https://ossci-windows.s3.amazonaws.com/%CUDA_INSTALL_EXE%" --output "%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%"
    set "ARGS=thrust_11.3 nvcc_11.3 cuobjdump_11.3 nvprune_11.3 nvprof_11.3 cupti_11.3 cublas_11.3 cublas_dev_11.3 cudart_11.3 cufft_11.3 cufft_dev_11.3 curand_11.3 curand_dev_11.3 cusolver_11.3 cusolver_dev_11.3 cusparse_11.3 cusparse_dev_11.3 npp_11.3 npp_dev_11.3 nvjpeg_11.3 nvjpeg_dev_11.3 nvrtc_11.3 nvrtc_dev_11.3 nvml_dev_11.3"

)

set CUDNN_INSTALL_ZIP=cudnn-11.3-windows-x64-v8.2.0.53.zip
if not exist "%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%" (
    curl -k -L "http://s3.amazonaws.com/ossci-windows/%CUDNN_INSTALL_ZIP%" --output "%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%"
)

goto cuda_common

:cuda115

set CUDA_INSTALL_EXE=cuda_11.5.0_496.13_win10.exe
if not exist "%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%" (
    curl -k -L "https://ossci-windows.s3.amazonaws.com/%CUDA_INSTALL_EXE%" --output "%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%"
    set "ARGS=thrust_11.5 nvcc_11.5 cuobjdump_11.5 nvprune_11.5 nvprof_11.5 cupti_11.5 cublas_11.5 cublas_dev_11.5 cudart_11.5 cufft_11.5 cufft_dev_11.5 curand_11.5 curand_dev_11.5 cusolver_11.5 cusolver_dev_11.5 cusparse_11.5 cusparse_dev_11.5 npp_11.5 npp_dev_11.5 nvrtc_11.5 nvrtc_dev_11.5 nvml_dev_11.5"
)

set CUDNN_INSTALL_ZIP=cudnn-11.3-windows-x64-v8.2.0.53.zip
if not exist "%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%" (
    curl -k -L "http://s3.amazonaws.com/ossci-windows/%CUDNN_INSTALL_ZIP%" --output "%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%"
)

goto cuda_common

:cuda_common

if not exist "%SRC_DIR%\temp_build\NvToolsExt.7z" (
    curl -k -L https://www.dropbox.com/s/9mcolalfdj4n979/NvToolsExt.7z?dl=1 --output "%SRC_DIR%\temp_build\NvToolsExt.7z"
    if errorlevel 1 exit /b 1
)

echo Installing CUDA toolkit...
7z x %CUDA_SETUP_FILE% -o"%SRC_DIR%\temp_build\cuda"
pushd "%SRC_DIR%\temp_build\cuda"
sc config wuauserv start= disabled
sc stop wuauserv
sc query wuauserv

start /wait setup.exe -s %ARGS% -loglevel:6 -log:"%cd%/cuda_install_logs"
echo %errorlevel%

popd

echo Installing VS integration...
rem It's for VS 2019
if "%CUDA_VER_MAJOR%" == "10" (
    xcopy /Y "%SRC_DIR%\temp_build\cuda\CUDAVisualStudioIntegration\extras\visual_studio_integration\MSBuildExtensions\*.*" "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations"
)
if "%CUDA_VER_MAJOR%" == "11" (
    xcopy /Y "%SRC_DIR%\temp_build\cuda\visual_studio_integration\CUDAVisualStudioIntegration\extras\visual_studio_integration\MSBuildExtensions\*.*" "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations"
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
set "NVTOOLSEXT_PATH=%ProgramFiles%\NVIDIA Corporation\NvToolsExt\bin\x64"

if not exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\bin\nvcc.exe" (
    echo CUDA %CUDA_VERSION_STR% installed failed.
    echo --------- RunDll32.exe.log
    type "%SRC_DIR%\temp_build\cuda\cuda_install_logs\LOG.RunDll32.exe.log"
    echo --------- setup.exe.log -------
    type "%SRC_DIR%\temp_build\cuda\cuda_install_logs\LOG.setup.exe.log"
    exit /b 1
)

echo Installing cuDNN...
7z x %CUDNN_SETUP_FILE% -o"%SRC_DIR%\temp_build\cudnn"
xcopy /Y "%SRC_DIR%\temp_build\cudnn\cuda\bin\*.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\bin"
xcopy /Y "%SRC_DIR%\temp_build\cudnn\cuda\lib\x64\*.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\lib\x64"
xcopy /Y "%SRC_DIR%\temp_build\cudnn\cuda\include\*.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\include"

echo Cleaning temp files
rd /s /q "%SRC_DIR%\temp_build" || ver > nul
