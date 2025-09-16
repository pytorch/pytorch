if "%BUILD_ENVIRONMENT%"=="" (
  set CONDA_PARENT_DIR=%CD%
) else (
  set CONDA_PARENT_DIR=C:\Jenkins
)
set CONDA_ROOT_DIR=%CONDA_PARENT_DIR%\Miniconda3

:: Be conservative here when rolling out the new AMI with conda. This will try
:: to install conda as before if it couldn't find the conda installation. This
:: can be removed eventually after we gain enough confidence in the AMI
if not exist %CONDA_ROOT_DIR% (
  set INSTALL_FRESH_CONDA=1
)

if "%INSTALL_FRESH_CONDA%"=="1" (
  curl --retry 3 --retry-all-errors -k https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe --output %TMP_DIR_WIN%\Miniconda3-latest-Windows-x86_64.exe
  if errorlevel 1 exit /b
  if not errorlevel 0 exit /b

  %TMP_DIR_WIN%\Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /AddToPath=0 /D=%CONDA_ROOT_DIR%
  if errorlevel 1 exit /b
  if not errorlevel 0 exit /b
)

:: Activate conda so that we can use its commands, i.e. conda, python, pip
call %CONDA_ROOT_DIR%\Scripts\activate.bat %CONDA_ROOT_DIR%
:: Activate conda so that we can use its commands, i.e. conda, python, pip
call conda activate py_tmp

call pip install -r requirements.txt
