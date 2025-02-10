@echo off

echo Dependency OpenBLAS installation started.

:: Pre-check for downloads and dependencies folders
if not exist "%DOWNLOADS_DIR%" mkdir %DOWNLOADS_DIR%
if not exist "%DEPENDENCIES_DIR%" mkdir %DEPENDENCIES_DIR%

:: activate visual studio
call "%DEPENDENCIES_DIR%\VSBuildTools\VC\Auxiliary\Build\vcvarsall.bat" arm64
where cl.exe

:: Clone OpenBLAS
cd %DEPENDENCIES_DIR%
git clone https://github.com/OpenMathLib/OpenBLAS.git -b v0.3.29

echo Configuring OpenBLAS...
mkdir OpenBLAS\build
cd OpenBLAS\build
cmake .. -G Ninja ^
  -DBUILD_TESTING=0 ^
  -DBUILD_BENCHMARKS=0 ^
  -DC_LAPACK=1 ^
  -DNOFORTRAN=1 ^
  -DDYNAMIC_ARCH=0 ^
  -DARCH=arm64 ^
  -DBINARY=64 ^
  -DTARGET=GENERIC ^
  -DUSE_OPENMP=1 ^
  -DCMAKE_SYSTEM_PROCESSOR=ARM64 ^
  -DCMAKE_SYSTEM_NAME=Windows ^
  -DCMAKE_BUILD_TYPE=Release

echo Building OpenBLAS...
cmake --build . --config Release

echo Installing OpenBLAS...
cmake --install . --prefix ../install

:: Check if installation was successful
if %errorlevel% neq 0 (
    echo "Failed to install OpenBLAS. (exitcode = %errorlevel%)"
    exit /b 1
)

echo Dependency OpenBLAS installation finished.