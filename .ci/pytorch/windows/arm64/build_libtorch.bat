@echo on

:: Set the CMAKE_BUILD_TYPE
set "CMAKE_BUILD_TYPE=%BUILD_TYPE%"

:: activate visual studio
call "%DEPENDENCIES_DIR%\VSBuildTools\VC\Auxiliary\Build\vcvarsall.bat" arm64 -vcvars_ver=%MSVC_VERSION%
where cl.exe

:: change to source directory
cd %PYTORCH_ROOT%

:: create virtual environment
python -m venv .venv
echo * > .venv\.gitignore
call .\.venv\Scripts\activate
where python

:: python install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

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