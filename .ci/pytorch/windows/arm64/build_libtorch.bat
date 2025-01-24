@echo on

:: environment variables
set CMAKE_BUILD_TYPE=%BUILD_TYPE%
set CMAKE_C_COMPILER_LAUNCHER=sccache
set CMAKE_CXX_COMPILER_LAUNCHER=sccache
set libuv_ROOT=%DEPENDENCIES_DIR%\libuv\install
set MSSdk=1
if defined PYTORCH_BUILD_VERSION (
  set PYTORCH_BUILD_VERSION=%PYTORCH_BUILD_VERSION%
  set PYTORCH_BUILD_NUMBER=1
)

:: Set BLAS type
if %ENABLE_APL% == 1 (
    set BLAS=APL
    set USE_LAPACK=1
) else if %ENABLE_OPENBLAS% == 1 (
    set BLAS=OpenBLAS
    set OpenBLAS_HOME=%DEPENDENCIES_DIR%\OpenBLAS\install
)

:: activate visual studio
call "%DEPENDENCIES_DIR%\VSBuildTools\VC\Auxiliary\Build\vcvarsall.bat" arm64
where cl.exe

:: change to source directory
cd %PYTORCH_ROOT%

:: copy libuv.dll
copy %libuv_ROOT%\lib\Release\uv.dll torch\lib\uv.dll

:: create virtual environment
python -m venv .venv
echo * > .venv\.gitignore
call .\.venv\Scripts\activate
where python

:: python install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
:: DISTUTILS_USE_SDK should be set after psutil dependency
set DISTUTILS_USE_SDK=1

:: start sccache server and reset sccache stats
sccache --start-server
sccache --zero-stats
sccache --show-stats
 
:: Prepare the environment
mkdir libtorch
mkdir libtorch\bin
mkdir libtorch\cmake
mkdir libtorch\include
mkdir libtorch\lib
mkdir libtorch\share
mkdir libtorch\test

:: Call LibTorch build script
python ./tools/build_libtorch.py

:: Check if there is an error
IF ERRORLEVEL 1 exit /b 1
IF NOT ERRORLEVEL 0 exit /b 1
 
:: Move the files to the correct location
move /Y torch\bin\*.* libtorch\bin\
move /Y torch\cmake\*.* libtorch\cmake\
robocopy /move /e torch\include\ libtorch\include\
move /Y torch\lib\*.* libtorch\lib\
robocopy /move /e torch\share\ libtorch\share\
move /Y torch\test\*.* libtorch\test\
move /Y libtorch\bin\*.dll libtorch\lib\

:: Set version
echo %PYTORCH_BUILD_VERSION% > libtorch\build-version
git rev-parse HEAD > libtorch\build-hash

:: Set LIBTORCH_PREFIX
IF "%DEBUG%" == "" (
    set LIBTORCH_PREFIX=libtorch-win-arm64-shared-with-deps
) ELSE (
    set LIBTORCH_PREFIX=libtorch-win-arm64-shared-with-deps-debug
)

:: Create output
C:\Windows\System32\tar.exe -cvaf %LIBTORCH_PREFIX%-%PYTORCH_BUILD_VERSION%.zip -C libtorch *

:: Copy output to target directory
if not exist ..\output mkdir ..\output
copy /Y "%LIBTORCH_PREFIX%-%PYTORCH_BUILD_VERSION%.zip" "%PYTORCH_FINAL_PACKAGE_DIR%\"
copy /Y "%LIBTORCH_PREFIX%-%PYTORCH_BUILD_VERSION%.zip" "%PYTORCH_FINAL_PACKAGE_DIR%\%LIBTORCH_PREFIX%-latest.zip"

:: Cleanup raw data to save space
rmdir /s /q libtorch

:: Check if installation was successful
if %errorlevel% neq 0 (
    echo "Failed on build_libtorch. (exitcode = %errorlevel%)"
    exit /b 1
)