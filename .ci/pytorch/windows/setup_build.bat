IF "%DESIRED_PYTHON%"=="" (
    echo DESIRED_PYTHON is NOT defined.
    exit /b 1
)

call "internal\install_python.bat"

%PYTHON_EXEC% --version
set "PATH=%CD%\Python\Lib\site-packages\cmake\data\bin;%CD%\Python\Scripts;%CD%\Python;%PATH%"
if "%DESIRED_PYTHON%" == "3.13t" %PYTHON_EXEC% -m pip install numpy==2.2.1 cmake
if "%DESIRED_PYTHON%" == "3.13" %PYTHON_EXEC% -m pip install numpy==2.1.2 cmake
if "%DESIRED_PYTHON%" == "3.12" %PYTHON_EXEC% -m pip install numpy==2.0.2 cmake
if "%DESIRED_PYTHON%" == "3.11" %PYTHON_EXEC% -m pip install numpy==2.0.2 cmake
if "%DESIRED_PYTHON%" == "3.10" %PYTHON_EXEC% -m pip install numpy==2.0.2 cmake
if "%DESIRED_PYTHON%" == "3.9" %PYTHON_EXEC% -m pip install numpy==2.0.2 cmake

%PYTHON_EXEC% -m pip install pyyaml
%PYTHON_EXEC% -m pip install mkl-include mkl-static
%PYTHON_EXEC% -m pip install boto3 ninja typing_extensions setuptools==72.1.0

where cmake.exe

:: Install libuv
curl -k https://s3.amazonaws.com/ossci-windows/libuv-1.40.0-h8ffe710_0.tar.bz2 -o libuv-1.40.0-h8ffe710_0.tar.bz2
7z x -aoa libuv-1.40.0-h8ffe710_0.tar.bz2
tar -xvf libuv-1.40.0-h8ffe710_0.tar -C %CD%\Python\
set libuv_ROOT=%CD%\Python\Library
