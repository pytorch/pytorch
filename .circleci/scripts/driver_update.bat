curl --retry 3 -kL "https://s3.amazonaws.com/ossci-windows/NVIDIA_Corporation.cer" --output "NVIDIA_Corporation.cer"
if errorlevel 1 exit /b 1

certutil -addstore "TrustedPublisher" NVIDIA_Corporation.cer
if errorlevel 1 exit /b 1

del NVIDIA_Corporation.cer
if errorlevel 1 exit /b 1

set "DRIVER_DOWNLOAD_LINK=https://s3.amazonaws.com/ossci-windows/451.82-tesla-desktop-winserver-2019-2016-international.exe"
curl --retry 3 -kL %DRIVER_DOWNLOAD_LINK% --output 451.82-tesla-desktop-winserver-2019-2016-international.exe
if errorlevel 1 exit /b 1

mkdir logs

start /wait 451.82-tesla-desktop-winserver-2019-2016-international.exe -s -noreboot -loglevel:6 -log:"%CD%:\logs"
if errorlevel 1 exit /b 1

type logs\LOG.setup.exe.log

rmdir /s /q logs

del 451.82-tesla-desktop-winserver-2019-2016-international.exe || ver > NUL

if not exist "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe" exit /b 1
