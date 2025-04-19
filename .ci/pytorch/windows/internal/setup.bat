@echo off

echo The flags after configuring:
echo USE_CUDA=%USE_CUDA%
echo CMAKE_GENERATOR=%CMAKE_GENERATOR%
if "%USE_CUDA%"==""  echo CUDA_PATH=%CUDA_PATH%
if NOT "%CC%"==""   echo CC=%CC%
if NOT "%CXX%"==""  echo CXX=%CXX%
if NOT "%DISTUTILS_USE_SDK%"==""  echo DISTUTILS_USE_SDK=%DISTUTILS_USE_SDK%

set SRC_DIR=%NIGHTLIES_PYTORCH_ROOT%

IF "%VSDEVCMD_ARGS%" == "" (
    call "%VS15VCVARSALL%" x64
) ELSE (
    call "%VS15VCVARSALL%" x64 %VSDEVCMD_ARGS%
)

pushd %SRC_DIR%

IF NOT exist "setup.py" (
    cd %MODULE_NAME%
)

if "%CXX%"=="sccache cl" goto sccache_start
if "%CXX%"=="sccache-cl" goto sccache_start
goto sccache_end

:sccache_start
set SCCACHE_IDLE_TIMEOUT=0

sccache --stop-server
sccache --start-server
sccache --zero-stats

:sccache_end


if "%BUILD_PYTHONLESS%" == "" goto pytorch else goto libtorch

:libtorch
set VARIANT=shared-with-deps

mkdir libtorch
mkdir libtorch\bin
mkdir libtorch\cmake
mkdir libtorch\include
mkdir libtorch\lib
mkdir libtorch\share
mkdir libtorch\test

mkdir build
pushd build
python ../tools/build_libtorch.py
popd

IF ERRORLEVEL 1 exit /b 1
IF NOT ERRORLEVEL 0 exit /b 1

move /Y torch\bin\*.* libtorch\bin\
move /Y torch\cmake\*.* libtorch\cmake\
robocopy /move /e torch\include\ libtorch\include\
move /Y torch\lib\*.* libtorch\lib\
robocopy /move /e torch\share\ libtorch\share\
move /Y torch\test\*.* libtorch\test\

move /Y libtorch\bin\*.dll libtorch\lib\

echo %PYTORCH_BUILD_VERSION% > libtorch\build-version
git rev-parse HEAD > libtorch\build-hash

IF "%DEBUG%" == "" (
    set LIBTORCH_PREFIX=libtorch-win-%VARIANT%
) ELSE (
    set LIBTORCH_PREFIX=libtorch-win-%VARIANT%-debug
)

7z a -tzip "%LIBTORCH_PREFIX%-%PYTORCH_BUILD_VERSION%.zip" libtorch\*
:: Cleanup raw data to save space
rmdir /s /q libtorch

if not exist ..\output mkdir ..\output
copy /Y "%LIBTORCH_PREFIX%-%PYTORCH_BUILD_VERSION%.zip" "%PYTORCH_FINAL_PACKAGE_DIR%\"
copy /Y "%LIBTORCH_PREFIX%-%PYTORCH_BUILD_VERSION%.zip" "%PYTORCH_FINAL_PACKAGE_DIR%\%LIBTORCH_PREFIX%-latest.zip"

goto build_end

:pytorch
python setup.py bdist_wheel -d "%PYTORCH_FINAL_PACKAGE_DIR%"

:build_end
IF ERRORLEVEL 1 exit /b 1
IF NOT ERRORLEVEL 0 exit /b 1

if "%CXX%"=="sccache cl" goto sccache_cleanup
if "%CXX%"=="sccache-cl" goto sccache_cleanup
goto sccache_cleanup_end

:sccache_cleanup
sccache --show-stats
taskkill /im sccache.exe /f /t || ver > nul
taskkill /im nvcc.exe /f /t || ver > nul

:sccache_cleanup_end

cd ..
