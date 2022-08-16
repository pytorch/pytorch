if "%BUILD_ENVIRONMENT%"=="" (
  set CONDA_PARENT_DIR=%CD%
) else (
  set CONDA_PARENT_DIR=C:\Jenkins
)


:: Be conservative here when rolling out the new AMI with conda. This will try
:: to install conda as before if it couldn't find the conda installation. This
:: can be removed eventually after we gain enough confidence in the AMI
if not exist %CONDA_PARENT_DIR%\Miniconda3 (
  set INSTALL_FRESH_CONDA=1
)

if "%INSTALL_FRESH_CONDA%"=="1" (
  curl --retry 3 -k https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe --output %TMP_DIR_WIN%\Miniconda3-latest-Windows-x86_64.exe
  if errorlevel 1 exit /b
  if not errorlevel 0 exit /b

  %TMP_DIR_WIN%\Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /AddToPath=0 /D=%CONDA_PARENT_DIR%\Miniconda3
  if errorlevel 1 exit /b
  if not errorlevel 0 exit /b
)

:: Activate conda so that we can use its commands, i.e. conda, python, pip
call %CONDA_PARENT_DIR%\Miniconda3\Scripts\activate.bat %CONDA_PARENT_DIR%\Miniconda3

if "%INSTALL_FRESH_CONDA%"=="1" (
  call conda install -y -q numpy"<1.23" cffi pyyaml boto3 libuv
  if errorlevel 1 exit /b
  if not errorlevel 0 exit /b

  call conda install -y -q -c conda-forge cmake=3.22.3
  if errorlevel 1 exit /b
  if not errorlevel 0 exit /b
)
