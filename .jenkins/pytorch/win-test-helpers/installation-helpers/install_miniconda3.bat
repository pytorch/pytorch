if "%BUILD_ENVIRONMENT%"=="" (
  set CONDA_PARENT_DIR=%CD%
) else (
  set CONDA_PARENT_DIR=C:\Jenkins
)

if "%REBUILD%"=="" set INSTALL_FRESH_CONDA=1
if NOT "%BUILD_ENVIRONMENT%"==""  set INSTALL_FRESH_CONDA=1

if "%INSTALL_FRESH_CONDA%"=="1" (
  IF EXIST %CONDA_PARENT_DIR%\Miniconda3 ( rd /s /q %CONDA_PARENT_DIR%\Miniconda3 )
  curl --retry 3 -k https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe --output %TMP_DIR_WIN%\Miniconda3-latest-Windows-x86_64.exe
  if errorlevel 1 exit /b
  if not errorlevel 0 exit /b
  %TMP_DIR_WIN%\Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /AddToPath=0 /D=%CONDA_PARENT_DIR%\Miniconda3
  if errorlevel 1 exit /b
  if not errorlevel 0 exit /b
)

call %CONDA_PARENT_DIR%\Miniconda3\Scripts\activate.bat %CONDA_PARENT_DIR%\Miniconda3
if "%INSTALL_FRESH_CONDA%"=="1" (
  call conda install -y -q python=%PYTHON_VERSION% numpy cffi pyyaml boto3 libuv
  if errorlevel 1 exit /b
  if not errorlevel 0 exit /b
  call conda install -y -q -c conda-forge cmake=3.22.3
  if errorlevel 1 exit /b
  if not errorlevel 0 exit /b
)
