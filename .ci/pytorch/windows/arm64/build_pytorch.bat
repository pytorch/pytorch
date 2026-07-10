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
    set USE_LAPACK=1
) else if %ENABLE_OPENBLAS% == 1 (
    set BLAS=OpenBLAS
    set OpenBLAS_HOME=%DEPENDENCIES_DIR%\OpenBLAS\install
)

:: activate visual studio
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" arm64
where cl.exe

if "%USE_CUDA%" == "1" (
    set USE_CUDNN=1
    if not defined CUDA_PATH set "CUDA_PATH=%CUDA_HOME%"
    set "CUDA_HOME=%CUDA_PATH%"
    set "CUDACXX=%CUDA_PATH%\bin\nvcc.exe"
    set "CUDNN_ROOT_DIR=%CUDNN_HOME%"
    set "CUDNN_INCLUDE_DIR=%CUDNN_HOME%\include"
    set "CUDNN_LIB_DIR=%CUDNN_HOME%\lib\arm64"
    set TORCH_CUDA_ARCH_LIST=TODO
    set CMAKE_CUDA_ARCHITECTURES=TODO
    set USE_MAGMA=1
    rem TH_BINARY_BUILD=1 links BLAS (ArmPL, which also carries LAPACK) INTO torch_cuda so magma's
    rem Fortran BLAS/LAPACK externals resolve (aten/src/ATen/CMakeLists.txt); else torch_cuda LNK2019.
    set TH_BINARY_BUILD=1
    set "CMAKE_CUDA_FLAGS=%CMAKE_CUDA_FLAGS% -Xcompiler /Zc:preprocessor"
    set "CFLAGS=/Zc:preprocessor /EHsc"
    set "CXXFLAGS=/Zc:preprocessor /EHsc"
)

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
