@echo off
setlocal

set "ORIG_PATH=%PATH%"

if "%PACKAGE_TYPE%" == "wheel" goto wheel
if "%PACKAGE_TYPE%" == "libtorch" goto libtorch

echo "unknown package type"
exit /b 1

:wheel
echo "install wheel package"

echo Running pip install...
pip install -q --pre numpy protobuf
echo Error level after pip install: %ERRORLEVEL%
if errorlevel 1 exit /b 1

for /F "delims=" %%i in ('where /R "%PYTORCH_FINAL_PACKAGE_DIR:/=\%" *.whl') do pip install "%%i"
if errorlevel 1 exit /b 1

goto smoke_test

:smoke_test
python -c "import torch"
if ERRORLEVEL 1 exit /b 1

echo Running python rnn_smoke.py...
python %PYTORCH_ROOT%\.ci\pytorch\test_example_code\rnn_smoke_win_arm64.py
if errorlevel 1 exit /b 1

echo Checking that basic CNN works...
python %PYTORCH_ROOT%\.ci\pytorch\test_example_code\cnn_smoke_win_arm64.py
if errorlevel 1 exit /b 1

goto end

:libtorch
echo "install and test libtorch"

for /F "delims=" %%i in ('where /R "%PYTORCH_FINAL_PACKAGE_DIR:/=\%" *-latest.zip') do tar -xf "%%i" -C tmp
if ERRORLEVEL 1 exit /b 1

pushd tmp\libtorch

set VC_VERSION_LOWER=14
set VC_VERSION_UPPER=36

call "%DEPENDENCIES_DIR%\VSBuildTools\VC\Auxiliary\Build\vcvarsall.bat" arm64

set install_root=%CD%
set INCLUDE=%INCLUDE%;%install_root%\include;%install_root%\include\torch\csrc\api\include
set LIB=%LIB%;%install_root%\lib
set PATH=%PATH%;%install_root%\lib

cl %PYTORCH_ROOT%\.ci\pytorch\test_example_code\simple-torch-test.cpp c10.lib torch_cpu.lib /EHsc /std:c++17
if ERRORLEVEL 1 exit /b 1

.\simple-torch-test.exe
if ERRORLEVEL 1 exit /b 1

:end
set "PATH=%ORIG_PATH%"
popd