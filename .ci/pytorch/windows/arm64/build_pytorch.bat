@echo on

:: environment variables
set CMAKE_BUILD_TYPE=%BUILD_TYPE%
set CMAKE_C_COMPILER_LAUNCHER=sccache
set CMAKE_CXX_COMPILER_LAUNCHER=sccache
set libuv_ROOT=%DEPENDENCIES_DIR%\libuv\install
set INSTALL_TEST=0
set MSSdk=1
if defined PYTORCH_BUILD_VERSION (
  set PYTORCH_BUILD_VERSION=%PYTORCH_BUILD_VERSION%
  set PYTORCH_BUILD_NUMBER=1
)

:: Set BLAS type
if %ENABLE_APL% == 1 (
    set BLAS=APL
) else if %ENABLE_OPENBLAS% == 1 (
    set BLAS=OpenBLAS
    set OpenBLAS_HOME=%DEPENDENCIES_DIR%\OpenBLAS\install
)

:: activate visual studio
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -find VC\Auxiliary\Build\vcvarsall.bat`) do call "%%i" arm64
where cl.exe

:: change to source directory
cd %PYTORCH_ROOT%

:: copy libuv.dll (cmake installs the dll to bin/, not lib/Release/)
copy %libuv_ROOT%\bin\uv.dll torch\lib\uv.dll

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
python -m build --wheel --no-isolation --outdir "%PYTORCH_FINAL_PACKAGE_DIR%"

:: show sccache stats
sccache --show-stats

:: Check if installation was successful
if %errorlevel% neq 0 (
    echo "Failed on build_pytorch. (exitcode = %errorlevel%)"
    exit /b 1
)
