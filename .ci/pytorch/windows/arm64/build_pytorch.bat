@echo on

:: environment variables
set BLAS=APL
set USE_LAPACK=1
set CMAKE_BUILD_TYPE=%BUILD_TYPE%
set CMAKE_C_COMPILER_LAUNCHER=sccache
set CMAKE_CXX_COMPILER_LAUNCHER=sccache
if defined PYTORCH_BUILD_VERSION (
  set PYTORCH_BUILD_VERSION=%PYTORCH_BUILD_VERSION%
  set PYTORCH_BUILD_NUMBER=1
)

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
pip install wheel
pip install -r requirements.txt

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