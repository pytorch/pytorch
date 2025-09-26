IF "%DESIRED_PYTHON%"=="" (
    echo DESIRED_PYTHON is NOT defined.
    exit /b 1
)

call "internal\install_python.bat"

%PYTHON_EXEC% --version
set "PATH=%CD%\Python\Lib\site-packages\cmake\data\bin;%CD%\Python\Scripts;%CD%\Python;%PATH%"

set NUMPY_PINNED_VERSION=""
if "%DESIRED_PYTHON%" == "3.14t" set NUMPY_PINNED_VERSION="==2.3.2"
if "%DESIRED_PYTHON%" == "3.14" set NUMPY_PINNED_VERSION="==2.3.2"
if "%DESIRED_PYTHON%" == "3.13t" set NUMPY_PINNED_VERSION="==2.2.1"
if "%DESIRED_PYTHON%" == "3.13" set NUMPY_PINNED_VERSION="==2.1.2"
if "%DESIRED_PYTHON%" == "3.12" set NUMPY_PINNED_VERSION="==2.0.2"
if "%DESIRED_PYTHON%" == "3.11" set NUMPY_PINNED_VERSION="==2.0.2"
if "%DESIRED_PYTHON%" == "3.10" set NUMPY_PINNED_VERSION="==2.0.2"
if "%DESIRED_PYTHON%" == "3.9" set NUMPY_PINNED_VERSION="==2.0.2"

%PYTHON_EXEC% -m pip install "numpy%NUMPY_PINNED_VERSION%" -r "%PYTORCH_ROOT%\requirements-build.txt"
%PYTHON_EXEC% -m pip install mkl-include mkl-static
%PYTHON_EXEC% -m pip install boto3

where cmake.exe

:: Install libuv
curl -k https://s3.amazonaws.com/ossci-windows/libuv-1.40.0-h8ffe710_0.tar.bz2 -o libuv-1.40.0-h8ffe710_0.tar.bz2
7z x -aoa libuv-1.40.0-h8ffe710_0.tar.bz2
tar -xvf libuv-1.40.0-h8ffe710_0.tar -C %CD%\Python\
set libuv_ROOT=%CD%\Python\Library
