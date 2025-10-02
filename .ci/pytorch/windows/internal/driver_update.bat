set WIN_DRIVER_VN=580.88
set "DRIVER_DOWNLOAD_LINK=https://ossci-windows.s3.amazonaws.com/%WIN_DRIVER_VN%-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe" & REM @lint-ignore
curl --retry 3 -kL %DRIVER_DOWNLOAD_LINK% --output %WIN_DRIVER_VN%-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe
if errorlevel 1 exit /b 1

start /wait %WIN_DRIVER_VN%-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe -s -noreboot
if errorlevel 1 exit /b 1

del %WIN_DRIVER_VN%-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe || ver > NUL
