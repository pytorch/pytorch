@echo off
setlocal

set "ORIG_PATH=%PATH%"

if "%PACKAGE_TYPE%" == "wheel" goto wheel
if "%PACKAGE_TYPE%" == "libtorch" goto libtorch

echo "unknown package type"
exit /b 1

:wheel
:: change to source directory
cd %PYTORCH_ROOT%

:: activate visual studio
call "%DEPENDENCIES_DIR%\VSBuildTools\VC\Auxiliary\Build\vcvarsall.bat" arm64
where cl.exe

:: create virtual environment
python -m venv .venv
echo * > .venv\.gitignore
call .\.venv\Scripts\activate
where python

:: install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest numpy protobuf

:: find file name for pytorch wheel
echo Searching for PyTorch wheel...
for /f "delims=" %%f in ('dir /b "%PYTORCH_FINAL_PACKAGE_DIR%" ^| findstr "torch-"') do set "TORCH_WHEEL_FILENAME=%PYTORCH_FINAL_PACKAGE_DIR%\%%f"

echo Installing PyTorch wheel...
pip install %TORCH_WHEEL_FILENAME%

echo pip list
pip list 

echo pip show torch
python -m pip show torch

echo import sys
python -c "import sys; print(sys.path)"
if ERRORLEVEL 1 exit /b 1

echo Checking that torch is installed...
python -c "import torch"
if ERRORLEVEL 1 exit /b 1

echo Running python rnn_smoke.py...
python .\.ci\pytorch\test_example_code\rnn_smoke_win_arm64.py
if errorlevel 1 exit /b 1

echo Checking that basic CNN works...
python .\.ci\pytorch\test_example_code\cnn_smoke_win_arm64.py
if errorlevel 1 exit /b 1

cd test

set CORE_TEST_LIST=test_autograd.py test_nn.py test_torch.py

for /L %%i in (1,1,%1) do (
    for %%t in (%CORE_TEST_LIST%) do (
        echo Running test: %%t
        python %%t --verbose --save-xml --use-pytest -vvvv -rfEsxXP -p no:xdist
    )
)

goto end

:libtorch
echo "install and test libtorch"

if not exist tmp mkdir tmp

for /F "delims=" %%i in ('where /R "%PYTORCH_FINAL_PACKAGE_DIR:/=\%" *-latest.zip') do C:\Windows\System32\tar.exe -xf "%%i" -C tmp
if ERRORLEVEL 1 exit /b 1

pushd tmp

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