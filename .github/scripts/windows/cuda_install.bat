@echo on

if "%CUDA_VERSION%" == "cpu" (
    echo Skipping for CPU builds
    exit /b 0
)
if "%CUDA_VERSION%" == "xpu" (
    echo Skipping for XPU builds
    exit /b 0
)

set SRC_DIR=%~dp0\..

if not exist "%SRC_DIR%\temp_build" mkdir "%SRC_DIR%\temp_build"

set /a CUDA_VER=%CUDA_VERSION%
set CUDA_VER_MAJOR=%CUDA_VERSION:~0,-1%
set CUDA_VER_MINOR=%CUDA_VERSION:~-1,1%
set CUDA_VERSION_STR=%CUDA_VER_MAJOR%.%CUDA_VER_MINOR%
set CUDNN_FOLDER="cuda"
set CUDNN_LIB_FOLDER="lib\x64"

:: Skip all of this if we already have cuda installed
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\bin\nvcc.exe" goto set_cuda_env_vars

if %CUDA_VER% EQU 118 goto cuda118
if %CUDA_VER% EQU 121 goto cuda121
if %CUDA_VER% EQU 124 goto cuda124
if %CUDA_VER% EQU 126 goto cuda126

echo CUDA %CUDA_VERSION_STR% is not supported
exit /b 1

:cuda118

set CUDA_INSTALL_EXE=cuda_11.8.0_522.06_windows.exe
if not exist "%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%" (
    curl -k -L "https://ossci-windows.s3.amazonaws.com/%CUDA_INSTALL_EXE%" --output "%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%"
    set "ARGS=cuda_profiler_api_11.8 thrust_11.8 nvcc_11.8 cuobjdump_11.8 nvprune_11.8 nvprof_11.8 cupti_11.8 cublas_11.8 cublas_dev_11.8 cudart_11.8 cufft_11.8 cufft_dev_11.8 curand_11.8 curand_dev_11.8 cusolver_11.8 cusolver_dev_11.8 cusparse_11.8 cusparse_dev_11.8 npp_11.8 npp_dev_11.8 nvrtc_11.8 nvrtc_dev_11.8 nvml_dev_11.8 nvtx_11.8"
)

set CUDNN_FOLDER=cudnn-windows-x86_64-9.5.0.50_cuda11-archive
set CUDNN_LIB_FOLDER="lib"
set "CUDNN_INSTALL_ZIP=%CUDNN_FOLDER%.zip"
if not exist "%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%" (
    curl -k -L "http://s3.amazonaws.com/ossci-windows/%CUDNN_INSTALL_ZIP%" --output "%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%"
)

@REM cuDNN 8.3+ required zlib to be installed on the path
echo Installing ZLIB dlls
curl -k -L "http://s3.amazonaws.com/ossci-windows/zlib123dllx64.zip" --output "%SRC_DIR%\temp_build\zlib123dllx64.zip"
7z x "%SRC_DIR%\temp_build\zlib123dllx64.zip" -o"%SRC_DIR%\temp_build\zlib"
xcopy /Y "%SRC_DIR%\temp_build\zlib\dll_x64\*.dll" "C:\Windows\System32"

goto cuda_common

:cuda121

set CUDA_INSTALL_EXE=cuda_12.1.1_531.14_windows.exe
if not exist "%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%" (
    curl -k -L "https://ossci-windows.s3.amazonaws.com/%CUDA_INSTALL_EXE%" --output "%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%"
    set "ARGS=cuda_profiler_api_12.1 thrust_12.1 nvcc_12.1 cuobjdump_12.1 nvprune_12.1 nvprof_12.1 cupti_12.1 cublas_12.1 cublas_dev_12.1 cudart_12.1 cufft_12.1 cufft_dev_12.1 curand_12.1 curand_dev_12.1 cusolver_12.1 cusolver_dev_12.1 cusparse_12.1 cusparse_dev_12.1 npp_12.1 npp_dev_12.1 nvrtc_12.1 nvrtc_dev_12.1 nvml_dev_12.1 nvjitlink_12.1 nvtx_12.1"
)

set CUDNN_FOLDER=cudnn-windows-x86_64-9.5.0.50_cuda12-archive
set CUDNN_LIB_FOLDER="lib"
set "CUDNN_INSTALL_ZIP=%CUDNN_FOLDER%.zip"
if not exist "%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%" (
    curl -k -L "http://s3.amazonaws.com/ossci-windows/%CUDNN_INSTALL_ZIP%" --output "%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%"
)

@REM cuDNN 8.3+ required zlib to be installed on the path
echo Installing ZLIB dlls
curl -k -L "http://s3.amazonaws.com/ossci-windows/zlib123dllx64.zip" --output "%SRC_DIR%\temp_build\zlib123dllx64.zip"
7z x "%SRC_DIR%\temp_build\zlib123dllx64.zip" -o"%SRC_DIR%\temp_build\zlib"
xcopy /Y "%SRC_DIR%\temp_build\zlib\dll_x64\*.dll" "C:\Windows\System32"

goto cuda_common

:cuda124

set CUDA_INSTALL_EXE=cuda_12.4.0_551.61_windows.exe
if not exist "%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%" (
    curl -k -L "https://ossci-windows.s3.amazonaws.com/%CUDA_INSTALL_EXE%" --output "%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%"
    set "ARGS=cuda_profiler_api_12.4 thrust_12.4 nvcc_12.4 cuobjdump_12.4 nvprune_12.4 nvprof_12.4 cupti_12.4 cublas_12.4 cublas_dev_12.4 cudart_12.4 cufft_12.4 cufft_dev_12.4 curand_12.4 curand_dev_12.4 cusolver_12.4 cusolver_dev_12.4 cusparse_12.4 cusparse_dev_12.4 npp_12.4 npp_dev_12.4 nvrtc_12.4 nvrtc_dev_12.4 nvml_dev_12.4 nvjitlink_12.4 nvtx_12.4"
)

set CUDNN_FOLDER=cudnn-windows-x86_64-9.5.0.50_cuda12-archive
set CUDNN_LIB_FOLDER="lib"
set "CUDNN_INSTALL_ZIP=%CUDNN_FOLDER%.zip"
if not exist "%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%" (
    curl -k -L "http://s3.amazonaws.com/ossci-windows/%CUDNN_INSTALL_ZIP%" --output "%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%"
)

@REM cuDNN 8.3+ required zlib to be installed on the path
echo Installing ZLIB dlls
curl -k -L "http://s3.amazonaws.com/ossci-windows/zlib123dllx64.zip" --output "%SRC_DIR%\temp_build\zlib123dllx64.zip"
7z x "%SRC_DIR%\temp_build\zlib123dllx64.zip" -o"%SRC_DIR%\temp_build\zlib"
xcopy /Y "%SRC_DIR%\temp_build\zlib\dll_x64\*.dll" "C:\Windows\System32"

goto cuda_common

:cuda126

set CUDA_INSTALL_EXE=cuda_12.6.2_560.94_windows.exe
if not exist "%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%" (
    curl -k -L "https://ossci-windows.s3.amazonaws.com/%CUDA_INSTALL_EXE%" --output "%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%"
    set "ARGS=cuda_profiler_api_12.6 thrust_12.6 nvcc_12.6 cuobjdump_12.6 nvprune_12.6 nvprof_12.6 cupti_12.6 cublas_12.6 cublas_dev_12.6 cudart_12.6 cufft_12.6 cufft_dev_12.6 curand_12.6 curand_dev_12.6 cusolver_12.6 cusolver_dev_12.6 cusparse_12.6 cusparse_dev_12.6 npp_12.6 npp_dev_12.6 nvrtc_12.6 nvrtc_dev_12.6 nvml_dev_12.6 nvjitlink_12.6 nvtx_12.6"
)

set CUDNN_FOLDER=cudnn-windows-x86_64-9.5.0.50_cuda12-archive
set CUDNN_LIB_FOLDER="lib"
set "CUDNN_INSTALL_ZIP=%CUDNN_FOLDER%.zip"
if not exist "%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%" (
    curl -k -L "http://s3.amazonaws.com/ossci-windows/%CUDNN_INSTALL_ZIP%" --output "%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%"
)

@REM cuDNN 8.3+ required zlib to be installed on the path
echo Installing ZLIB dlls
curl -k -L "http://s3.amazonaws.com/ossci-windows/zlib123dllx64.zip" --output "%SRC_DIR%\temp_build\zlib123dllx64.zip"
7z x "%SRC_DIR%\temp_build\zlib123dllx64.zip" -o"%SRC_DIR%\temp_build\zlib"
xcopy /Y "%SRC_DIR%\temp_build\zlib\dll_x64\*.dll" "C:\Windows\System32"

goto cuda_common

:cuda_common
:: NOTE: We only install CUDA if we don't have it installed already.
:: With GHA runners these should be pre-installed as part of our AMI process
:: If you cannot find the CUDA version you want to build for here then please
:: add it @ https://github.com/pytorch/test-infra/tree/main/aws/ami/windows
if not exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\bin\nvcc.exe" (
    if not exist "%SRC_DIR%\temp_build\NvToolsExt.7z" (
        curl -k -L https://ossci-windows.s3.us-east-1.amazonaws.com/builder/NvToolsExt.7z --output "%SRC_DIR%\temp_build\NvToolsExt.7z"
        if errorlevel 1 exit /b 1
    )

    if not exist "%SRC_DIR%\temp_build\gpu_driver_dlls.zip" (
        curl -k -L "https://ossci-windows.s3.us-east-1.amazonaws.com/builder/additional_dlls.zip" --output "%SRC_DIR%\temp_build\gpu_driver_dlls.zip"
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
    if "%VC_YEAR%" == "2019" (
        xcopy /Y "%SRC_DIR%\temp_build\cuda\CUDAVisualStudioIntegration\extras\visual_studio_integration\MSBuildExtensions\*.*" "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Microsoft\VC\v160\BuildCustomizations"
    )
    if "%VC_YEAR%" == "2022" (
        xcopy /Y "%SRC_DIR%\temp_build\cuda\CUDAVisualStudioIntegration\extras\visual_studio_integration\MSBuildExtensions\*.*" "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations"
    )

    echo Installing NvToolsExt...
    7z x %SRC_DIR%\temp_build\NvToolsExt.7z -o"%SRC_DIR%\temp_build\NvToolsExt"
    mkdir "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\bin\x64"
    mkdir "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\include"
    mkdir "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\lib\x64"
    xcopy /Y "%SRC_DIR%\temp_build\NvToolsExt\bin\x64\*.*" "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\bin\x64"
    xcopy /Y "%SRC_DIR%\temp_build\NvToolsExt\include\*.*" "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\include"
    xcopy /Y "%SRC_DIR%\temp_build\NvToolsExt\lib\x64\*.*" "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\lib\x64"

    echo Installing cuDNN...
    7z x %CUDNN_SETUP_FILE% -o"%SRC_DIR%\temp_build\cudnn"
    xcopy /Y "%SRC_DIR%\temp_build\cudnn\%CUDNN_FOLDER%\bin\*.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\bin"
    xcopy /Y "%SRC_DIR%\temp_build\cudnn\%CUDNN_FOLDER%\%CUDNN_LIB_FOLDER%\*.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\lib\x64"
    xcopy /Y "%SRC_DIR%\temp_build\cudnn\%CUDNN_FOLDER%\include\*.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\include"

    echo Installing GPU driver DLLs
    7z x %SRC_DIR%\temp_build\gpu_driver_dlls.zip -o"C:\Windows\System32"

    echo Cleaning temp files
    rd /s /q "%SRC_DIR%\temp_build" || ver > nul

    if not exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\bin\nvcc.exe" (
        echo CUDA %CUDA_VERSION_STR% installed failed.
        echo --------- setup.exe.log -------
        type "%SRC_DIR%\temp_build\cuda\cuda_install_logs\LOG.setup.exe.log"
        echo --------- RunDll32.exe.log
        type "%SRC_DIR%\temp_build\cuda\cuda_install_logs\LOG.RunDll32.exe.log"
        exit /b 1
    )
)

goto set_cuda_env_vars

:set_cuda_env_vars

echo Setting up environment...
set "PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\bin;%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\libnvvp;%PATH%"
set "CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%"
set "CUDA_PATH_V%CUDA_VER_MAJOR%_%CUDA_VER_MINOR%=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%"
set "NVTOOLSEXT_PATH=%ProgramFiles%\NVIDIA Corporation\NvToolsExt"
