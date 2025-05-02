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

set ADDITIONAL_OPTIONS=""
set PYTHON_EXEC="python"
if "%DESIRED_PYTHON%" == "3.13t" (
    set ADDITIONAL_OPTIONS="Include_freethreaded=1"
    set PYTHON_EXEC="python3.13t"
)

del python-amd64.exe
curl --retry 3 -kL "%PYTHON_INSTALLER_URL%" --output python-amd64.exe
if errorlevel 1 exit /b 1

start /wait "" python-amd64.exe /quiet InstallAllUsers=1 PrependPath=0 Include_test=0 %ADDITIONAL_OPTIONS% TargetDir=%CD%\Python
if errorlevel 1 exit /b 1

set "PATH=%CD%\Python%DESIRED_PYTHON%\Scripts;%CD%\Python%DESIRED_PYTHON%;%PATH%"
