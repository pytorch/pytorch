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

:: Call PyTorch build script
python setup.py bdist_wheel -d "%PYTORCH_FINAL_PACKAGE_DIR%"

:: show sccache stats
sccache --show-stats

:: Check if installation was successful
if %errorlevel% neq 0 (
    echo "Failed on build_pytorch. (exitcode = %errorlevel%)"
    exit /b 1
)