@echo on
REM Description: Install Intel Support Packages on Windows
REM BKM reference: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html

if not "%CUDA_VERSION%" == "xpu" (
    echo Skipping for non XPU builds
    exit /b 0
)

set SRC_DIR=%NIGHTLIES_PYTORCH_ROOT%
if not exist "%SRC_DIR%\temp_build" mkdir "%SRC_DIR%\temp_build"

:xpu_bundle_install_start

set XPU_BUNDLE_PARENT_DIR=C:\Program Files (x86)\Intel\oneAPI
set XPU_BUNDLE_URL=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/9d6d6c17-ca2d-4735-9331-99447e4a1280/intel-deep-learning-essentials-2025.0.1.28_offline.exe
set XPU_BUNDLE_PRODUCT_NAME=intel.oneapi.win.deep-learning-essentials.product
set XPU_BUNDLE_VERSION=2025.0.1+20
set XPU_BUNDLE_INSTALLED=0
set XPU_BUNDLE_UNINSTALL=0
set XPU_EXTRA_URL=NULL
set XPU_EXTRA_PRODUCT_NAME=intel.oneapi.win.compiler.product
set XPU_EXTRA_VERSION=2025.0.1+1226
set XPU_EXTRA_INSTALLED=0
set XPU_EXTRA_UNINSTALL=0

if not [%XPU_VERSION%]==[] if [%XPU_VERSION%]==[2025.1] (
    set XPU_BUNDLE_URL=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/1a9fff3d-04c2-4d77-8861-3d86c774b66f/intel-deep-learning-essentials-2025.1.1.26_offline.exe
    set XPU_BUNDLE_VERSION=2025.1.1+23
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

if not "%XPU_ENABLE_KINETO%"=="1" goto install_end
:: Install Level Zero SDK
set XPU_EXTRA_LZ_URL=https://github.com/oneapi-src/level-zero/releases/download/v1.14.0/level-zero-sdk_1.14.0.zip
curl -k -L %XPU_EXTRA_LZ_URL% --output "%SRC_DIR%\temp_build\level_zero_sdk.zip"
echo "Installing level zero SDK..."
7z x "%SRC_DIR%\temp_build\level_zero_sdk.zip" -o"%SRC_DIR%\temp_build\level_zero"
set "INCLUDE=%SRC_DIR%\temp_build\level_zero\include;%INCLUDE%"
del "%SRC_DIR%\temp_build\level_zero_sdk.zip"

:install_end
