@echo on
REM Description: Install Intel Support Packages on Windows
REM BKM reference: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html

set XPU_INSTALL_MODE=%~1
if "%XPU_INSTALL_MODE%"=="" goto xpu_bundle_install_start
if "%XPU_INSTALL_MODE%"=="bundle" goto xpu_bundle_install_start
if "%XPU_INSTALL_MODE%"=="driver" goto xpu_driver_install_start
if "%XPU_INSTALL_MODE%"=="all" goto xpu_driver_install_start

:arg_error

echo Illegal XPU installation mode. The value can be "bundle"/"driver"/"all"
echo If keep the value as space, will use default "bundle" mode
exit /b 1

:xpu_driver_install_start
:: TODO Need more testing for driver installation
set XPU_DRIVER_LINK=https://downloadmirror.intel.com/830975/gfx_win_101.5972.exe
curl -o xpu_driver.exe --retry 3 --retry-all-errors -k %XPU_DRIVER_LINK%
echo "XPU Driver installing..."
start /wait "Intel XPU Driver Installer" "xpu_driver.exe"
if errorlevel 1 exit /b 1
del xpu_driver.exe
if "%XPU_INSTALL_MODE%"=="driver" goto xpu_install_end

:xpu_bundle_install_start

set XPU_BUNDLE_PARENT_DIR=C:\Program Files (x86)\Intel\oneAPI
set XPU_BUNDLE_URL=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/9d1a91e2-e8b8-40a5-8c7f-5db768a6a60c/w_intel-for-pytorch-gpu-dev_p_0.5.3.37_offline.exe
set XPU_BUNDLE_PRODUCT_NAME=intel.oneapi.win.intel-for-pytorch-gpu-dev.product
set XPU_BUNDLE_VERSION=0.5.3+31
set XPU_BUNDLE_INSTALLED=0
set XPU_BUNDLE_UNINSTALL=0
set XPU_EXTRA_URL=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/9d1a91e2-e8b8-40a5-8c7f-5db768a6a60c/w_intel-pti-dev_p_0.9.0.37_offline.exe
set XPU_EXTRA_PRODUCT_NAME=intel.oneapi.win.intel-pti-dev.product
set XPU_EXTRA_VERSION=0.9.0+36
set XPU_EXTRA_INSTALLED=0
set XPU_EXTRA_UNINSTALL=0

if not [%XPU_VERSION%]==[] if [%XPU_VERSION%]==[2025.0] (
    set XPU_BUNDLE_URL=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/efc86abd-cb77-452e-a03f-a741895b8ece/intel-deep-learning-essentials-2025.0.0.336_offline.exe
    set XPU_BUNDLE_PRODUCT_NAME=intel.oneapi.win.deep-learning-essentials.product
    set XPU_BUNDLE_VERSION=2025.0.0+335
    set XPU_BUNDLE_INSTALLED=0
    set XPU_BUNDLE_UNINSTALL=0
    set XPU_EXTRA_URL=NULL
    set XPU_EXTRA_PRODUCT_NAME=intel.oneapi.win.compiler.product
    set XPU_EXTRA_VERSION=2025.0.1+1226
    set XPU_EXTRA_INSTALLED=0
    set XPU_EXTRA_UNINSTALL=0
)

:: Check if XPU bundle is target version or already installed
if exist "%XPU_BUNDLE_PARENT_DIR%\Installer\installer.exe" goto xpu_bundle_ver_check
goto xpu_bundle_install

:xpu_bundle_ver_check

"%XPU_BUNDLE_PARENT_DIR%\Installer\installer.exe" --list-products > xpu_bundle_installed_ver.log

for /f "tokens=1,2" %%a in (xpu_bundle_installed_ver.log) do (
    if "%%a"=="%XPU_BUNDLE_PRODUCT_NAME%" (
        echo %%a Installed Version: %%b
        set XPU_BUNDLE_INSTALLED=1
        if not "%XPU_BUNDLE_VERSION%"=="%%b" (
            start /wait "Installer Title" "%XPU_BUNDLE_PARENT_DIR%\Installer\installer.exe" --action=remove --eula=accept --silent --product-id %%a --product-ver %%b --log-dir uninstall_bundle
            set XPU_BUNDLE_UNINSTALL=1
        )
    )
    if "%%a"=="%XPU_EXTRA_PRODUCT_NAME%" (
        echo %%a Installed Version: %%b
        set XPU_EXTRA_INSTALLED=1
        if not "%XPU_EXTRA_VERSION%"=="%%b" (
            start /wait "Installer Title" "%XPU_BUNDLE_PARENT_DIR%\Installer\installer.exe" --action=remove --eula=accept --silent --product-id %%a --product-ver %%b --log-dir uninstall_bundle
            set XPU_EXTRA_UNINSTALL=1
        )
    )
    if not "%%b" == "Version" if not [%%b]==[] if not "%%a"=="%XPU_BUNDLE_PRODUCT_NAME%" if not "%%a"=="%XPU_EXTRA_PRODUCT_NAME%" (
        echo "Uninstalling...."
        start /wait "Installer Title" "%XPU_BUNDLE_PARENT_DIR%\Installer\installer.exe" --action=remove --eula=accept --silent --product-id %%a --product-ver %%b --log-dir uninstall_bundle
    )
)
if errorlevel 1 exit /b 1
if exist xpu_bundle_installed_ver.log del xpu_bundle_installed_ver.log
if exist uninstall_bundle rmdir /s /q uninstall_bundle
if "%XPU_BUNDLE_INSTALLED%"=="0" goto xpu_bundle_install
if "%XPU_BUNDLE_UNINSTALL%"=="1" goto xpu_bundle_install

:xpu_extra_check

if "%XPU_EXTRA_URL%"=="NULL" goto xpu_install_end
if "%XPU_EXTRA_INSTALLED%"=="0" goto xpu_extra_install
if "%XPU_EXTRA_UNINSTALL%"=="1" goto xpu_extra_install
goto xpu_install_end

:xpu_bundle_install

curl -o xpu_bundle.exe --retry 3 --retry-all-errors -k %XPU_BUNDLE_URL%
echo "XPU Bundle installing..."
start /wait "Intel Pytorch Bundle Installer" "xpu_bundle.exe" --action=install --eula=accept --silent --log-dir install_bundle
if errorlevel 1 exit /b 1
del xpu_bundle.exe
goto xpu_extra_check

:xpu_extra_install

curl -o xpu_extra.exe --retry 3 --retry-all-errors -k %XPU_EXTRA_URL%
echo "Intel XPU EXTRA installing..."
start /wait "Intel XPU EXTRA Installer" "xpu_extra.exe" --action=install --eula=accept --silent --log-dir install_bundle
if errorlevel 1 exit /b 1
del xpu_extra.exe

:xpu_install_end
