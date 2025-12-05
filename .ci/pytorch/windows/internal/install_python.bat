set ADDITIONAL_OPTIONS=""
set PYTHON_EXEC="python"


if "%DESIRED_PYTHON%" == "3.13t" (
    echo Python version is set to 3.13t
    set "PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.13.0/python-3.13.0-amd64.exe"
    set ADDITIONAL_OPTIONS="Include_freethreaded=1"
    set PYTHON_EXEC="python3.13t"
) else if "%DESIRED_PYTHON%"=="3.14t" (
    echo Python version is set to 3.14 or 3.14t
    set "PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.14.0/python-3.14.0-amd64.exe"
    set ADDITIONAL_OPTIONS="Include_freethreaded=1"
    set PYTHON_EXEC="python3.14t"
) else (
    echo Python version is set to %DESIRED_PYTHON%
    set "PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/%DESIRED_PYTHON%.0/python-%DESIRED_PYTHON%.0-amd64.exe" %= @lint-ignore =%
)

del python-amd64.exe
curl --retry 3 -kL "%PYTHON_INSTALLER_URL%" --output python-amd64.exe
if errorlevel 1 exit /b 1

start /wait "" python-amd64.exe /quiet InstallAllUsers=1 PrependPath=0 Include_test=0 %ADDITIONAL_OPTIONS% TargetDir=%CD%\Python
if errorlevel 1 exit /b 1

set "PATH=%CD%\Python\Scripts;%CD%\Python;%PATH%"
%PYTHON_EXEC% -m pip install --upgrade pip setuptools packaging wheel build
if errorlevel 1 exit /b 1
