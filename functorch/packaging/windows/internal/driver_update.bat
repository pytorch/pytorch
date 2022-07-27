set "DRIVER_DOWNLOAD_LINK=https://ossci-windows.s3.amazonaws.com/461.09-data-center-tesla-desktop-winserver-2019-2016-international.exe"
curl --retry 3 -kL %DRIVER_DOWNLOAD_LINK% --output 461.09-data-center-tesla-desktop-winserver-2019-2016-international.exe
if errorlevel 1 exit /b 1

start /wait 461.09-data-center-tesla-desktop-winserver-2019-2016-international.exe -s -noreboot
if errorlevel 1 exit /b 1

del 461.09-data-center-tesla-desktop-winserver-2019-2016-international.exe || ver > NUL

setlocal EnableDelayedExpansion
set NVIDIA_GPU_EXISTS=0
for /F "delims=" %%i in ('wmic path win32_VideoController get name') do (
    set GPUS=%%i
    if not "x!GPUS:NVIDIA=!" == "x!GPUS!" (
        SET NVIDIA_GPU_EXISTS=1
        goto gpu_check_end
    )
)
:gpu_check_end
endlocal & set NVIDIA_GPU_EXISTS=%NVIDIA_GPU_EXISTS%

if "%NVIDIA_GPU_EXISTS%" == "0" (
    echo "CUDA Driver installation Failed"
    exit /b 1
)
