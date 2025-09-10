set SRC_DIR=%~dp0

pushd %SRC_DIR%\..

if not "%CUDA_VERSION%" == "cpu" if not "%CUDA_VERSION%" == "xpu" call internal\driver_update.bat
if errorlevel 1 exit /b 1

if "%CUDA_VERSION%" == "xpu" (
    call internal\xpu_install.bat
    if errorlevel 1 exit /b 1
    call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat"
    if errorlevel 1 exit /b 1
)

set "ORIG_PATH=%PATH%"

setlocal EnableDelayedExpansion
set NVIDIA_GPU_EXISTS=0
for /F "delims=" %%i in ('wmic path win32_VideoController get name') do (
    set GPUS=%%i
    if not "x!GPUS:NVIDIA=!" == "x!GPUS!" (
        SET NVIDIA_GPU_EXISTS=1
        goto gpu_check_end
    )
)
:gpu_check_end
endlocal & set NVIDIA_GPU_EXISTS=%NVIDIA_GPU_EXISTS%

if "%PACKAGE_TYPE%" == "wheel" goto wheel
if "%PACKAGE_TYPE%" == "libtorch" goto libtorch

echo "unknown package type"
exit /b 1

:wheel
echo "install wheel package"

call "internal\install_python.bat"

if "%DESIRED_PYTHON%" == "3.13t" %PYTHON_EXEC% -m pip install --pre numpy==2.2.1 protobuf
if "%DESIRED_PYTHON%" == "3.13" %PYTHON_EXEC% -m pip install --pre numpy==2.1.2 protobuf
if "%DESIRED_PYTHON%" == "3.12" %PYTHON_EXEC% -m pip install --pre numpy==2.0.2 protobuf
if "%DESIRED_PYTHON%" == "3.11" %PYTHON_EXEC% -m pip install --pre numpy==2.0.2 protobuf
if "%DESIRED_PYTHON%" == "3.10" %PYTHON_EXEC% -m pip install --pre numpy==2.0.2 protobuf
if "%DESIRED_PYTHON%" == "3.9" %PYTHON_EXEC% -m pip install --pre numpy==2.0.2 protobuf networkx

if errorlevel 1 exit /b 1

if "%PYTORCH_BUILD_VERSION:dev=%" NEQ "%PYTORCH_BUILD_VERSION%" (
    set "CHANNEL=nightly"
) else (
    set "CHANNEL=test"
)

set "EXTRA_INDEX= "
if "%CUDA_VERSION%" == "xpu" set "EXTRA_INDEX=--index-url https://download.pytorch.org/whl/%CHANNEL%/xpu"  %= @lint-ignore =%

for /F "delims=" %%i in ('where /R "%PYTORCH_FINAL_PACKAGE_DIR:/=\%" *.whl') do %PYTHON_EXEC% -m pip install "%%i" %EXTRA_INDEX%
if errorlevel 1 exit /b 1

goto smoke_test

:smoke_test
%PYTHON_EXEC% -c "import torch"
if ERRORLEVEL 1 exit /b 1

echo Checking that MKL is available
%PYTHON_EXEC% -c "import torch; exit(0 if torch.backends.mkl.is_available() else 1)"
if ERRORLEVEL 1 exit /b 1

if "%NVIDIA_GPU_EXISTS%" == "0" (
    echo "Skip CUDA tests for machines without a Nvidia GPU card"
    goto end
)

echo Checking that CUDA archs are setup correctly
%PYTHON_EXEC% -c "import torch; torch.randn([3,5]).cuda()"
if ERRORLEVEL 1 exit /b 1

echo Checking that magma is available
%PYTHON_EXEC% -c "import torch; torch.rand(1).cuda(); exit(0 if torch.cuda.has_magma else 1)"
if ERRORLEVEL 1 exit /b 1

echo Checking that CuDNN is available
%PYTHON_EXEC% -c "import torch; exit(0 if torch.backends.cudnn.is_available() else 1)"
if ERRORLEVEL 1 exit /b 1

echo Checking that basic RNN works
%PYTHON_EXEC% %PYTORCH_ROOT%\.ci\pytorch\test_example_code\rnn_smoke.py

if ERRORLEVEL 1 exit /b 1

echo Checking that basic CNN works
%PYTHON_EXEC% %PYTORCH_ROOT%\.ci\pytorch\test_example_code\cnn_smoke.py
if ERRORLEVEL 1 exit /b 1

goto end

:libtorch
echo "install and test libtorch"

if "%VC_YEAR%" == "2022" powershell internal\vs2022_install.ps1

if ERRORLEVEL 1 exit /b 1

for /F "delims=" %%i in ('where /R "%PYTORCH_FINAL_PACKAGE_DIR:/=\%" *-latest.zip') do 7z x "%%i" -otmp
if ERRORLEVEL 1 exit /b 1

pushd tmp\libtorch

set VC_VERSION_LOWER=17
set VC_VERSION_UPPER=18

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -legacy -products * -version [%VC_VERSION_LOWER%^,%VC_VERSION_UPPER%^) -property installationPath`) do (
    if exist "%%i" if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VS15INSTALLDIR=%%i"
        set "VS15VCVARSALL=%%i\VC\Auxiliary\Build\vcvarsall.bat"
        goto vswhere
    )
)

:vswhere
IF "%VS15VCVARSALL%"=="" (
    echo Visual Studio %VC_YEAR% C++ BuildTools is required to compile PyTorch test on Windows
    exit /b 1
)
call "%VS15VCVARSALL%" x64

set install_root=%CD%
set INCLUDE=%INCLUDE%;%install_root%\include;%install_root%\include\torch\csrc\api\include
set LIB=%LIB%;%install_root%\lib
set PATH=%PATH%;%install_root%\lib

cl %PYTORCH_ROOT%\.ci\pytorch\test_example_code\simple-torch-test.cpp c10.lib torch_cpu.lib /EHsc /std:c++17
if ERRORLEVEL 1 exit /b 1

.\simple-torch-test.exe
if ERRORLEVEL 1 exit /b 1

cl %PYTORCH_ROOT%\.ci\pytorch\test_example_code\check-torch-mkl.cpp c10.lib torch_cpu.lib /EHsc /std:c++17
if ERRORLEVEL 1 exit /b 1

.\check-torch-mkl.exe
if ERRORLEVEL 1 exit /b 1

if "%NVIDIA_GPU_EXISTS%" == "0" (
    echo "Skip CUDA tests for machines without a Nvidia GPU card"
    goto end
)

cl %PYTORCH_ROOT%\.ci\pytorch\test_example_code\check-torch-cuda.cpp torch_cpu.lib c10.lib torch_cuda.lib /EHsc /std:c++17 /link /INCLUDE:?warp_size@cuda@at@@YAHXZ
.\check-torch-cuda.exe
if ERRORLEVEL 1 exit /b 1

popd

echo Cleaning temp files
rd /s /q "tmp" || ver > nul

:end
set "PATH=%ORIG_PATH%"
popd
