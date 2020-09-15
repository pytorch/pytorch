if "%BUILD_ENVIRONMENT%"=="" (
  set CONDA_PARENT_DIR=%CD%
) else (
  set CONDA_PARENT_DIR=C:\Jenkins
)
if "%REBUILD%"=="" (
  IF EXIST %CONDA_PARENT_DIR%\Miniconda3 ( rd /s /q %CONDA_PARENT_DIR%\Miniconda3 )
  curl --retry 3 -k https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe --output %TMP_DIR_WIN%\Miniconda3-latest-Windows-x86_64.exe
  %TMP_DIR_WIN%\Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /AddToPath=0 /D=%CONDA_PARENT_DIR%\Miniconda3
)
call %CONDA_PARENT_DIR%\Miniconda3\Scripts\activate.bat %CONDA_PARENT_DIR%\Miniconda3
if "%REBUILD%"=="" (
  call conda install -y -q python=%PYTHON_VERSION% numpy cffi pyyaml boto3
  call conda install -y -q -c conda-forge cmake
  call conda install -y -q -c rdonnelly libuv
)

:: Get installed libuv path
@echo off
FOR /f "tokens=1,2,3* delims=: " %%a IN ('conda info') DO (
 IF "%%a %%b %%c"=="active env location" SET "active_conda_path=%%d\Library"
)
mkdir %active_conda_path%\lib\release
copy %active_conda_path%\bin\uv.dll %active_conda_path%\lib\release
copy %active_conda_path%\lib\uv.lib %active_conda_path%\lib\release
copy %active_conda_path%\lib\uv_a.lib %active_conda_path%\lib\release
set libuv_ROOT=%active_conda_path%
@echo on
echo libuv_ROOT=%active_conda_path%
