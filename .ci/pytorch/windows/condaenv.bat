IF "%DESIRED_PYTHON%"=="" (
    echo DESIRED_PYTHON is NOT defined.
    exit /b 1
)

echo Setting up Python %DESIRED_PYTHON%
set PYTHON_INSTALLER_URL=
if "%DESIRED_PYTHON%" == "3.13t" set "PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.13.0/python-3.13.0-amd64.exe"
if "%DESIRED_PYTHON%" == "3.13" set "PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.13.0/python-3.13.0-amd64.exe"
if "%DESIRED_PYTHON%" == "3.12" set "PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe"
if "%DESIRED_PYTHON%" == "3.11" set "PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe"
if "%DESIRED_PYTHON%" == "3.10" set "PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe"
if "%DESIRED_PYTHON%" == "3.9" set "PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.9.0/python-3.9.0-amd64.exe"
if "%PYTHON_INSTALLER_URL%" == "" (
    echo Python %DESIRED_PYTHON% not supported yet
)

del python-amd64.exe

echo Downloading Python %DESIRED_PYTHON%
curl --retry 3 -kL "%PYTHON_INSTALLER_URL%" --output python-amd64.exe
if errorlevel 1 exit /b 1

set ADDITIONAL_OPTIONS=""
if "%DESIRED_PYTHON%" == "3.13t" (
    set ADDITIONAL_OPTIONS="Include_freethreaded=1"
)

echo Installing Python %DESIRED_PYTHON%
start /wait "" python-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0 %ADDITIONAL_OPTIONS% TargetDir=%CD%\Python
if errorlevel 1 exit /b 1

:: Create venv and activate it
echo Activating Environment py%DESIRED_PYTHON%
py -%DESIRED_PYTHON% -mvenv py%DESIRED_PYTHON%
py%DESIRED_PYTHON%\Scripts\activate.bat

python --version

set "PATH=%CD%\Python\Lib\site-packages\cmake\data\bin;%CD%\Python\Scripts;%CD%\Python;%PATH%"
echo Installing numpy and cmake
if "%DESIRED_PYTHON%" == "3.13t" python -m pip install numpy==2.2.1 cmake
if "%DESIRED_PYTHON%" == "3.13" python -m pip install numpy==2.1.2 cmake
if "%DESIRED_PYTHON%" == "3.12" python -m pip install numpy==2.0.2 cmake
if "%DESIRED_PYTHON%" == "3.11" python -m pip install numpy==2.0.2 cmake
if "%DESIRED_PYTHON%" == "3.10" python -m pip install numpy==2.0.2 cmake
if "%DESIRED_PYTHON%" == "3.9"  python -m pip install numpy==2.0.2 cmake

python -m pip install pyyaml
python -m pip install mkl-include mkl-static
python -m pip install boto3 ninja typing_extensions setuptools==72.1.0

:: Install libuv
curl -k https://s3.amazonaws.com/ossci-windows/libuv-1.40.0-h8ffe710_0.tar.bz2 -o libuv-1.40.0-h8ffe710_0.tar.bz2
7z x -aoa libuv-1.40.0-h8ffe710_0.tar.bz2
tar -xvf libuv-1.40.0-h8ffe710_0.tar -C %CD%\Python%PYTHON_VERSION%\
set libuv_ROOT=%CD%\Python%PYTHON_VERSION%\Library
