if "%BUILD_ENVIRONMENT%"=="" (
  set CONDA_PARENT_DIR=%CD%
) else (
  set CONDA_PARENT_DIR=C:\Jenkins
)

set PATH=%CONDA_PARENT_DIR%\Miniconda3\Library\bin;%CONDA_PARENT_DIR%\Miniconda3;%CONDA_PARENT_DIR%\Miniconda3\Scripts;%PATH%

set tmp_dir_win=%1
set conda_parent_dir=%2

%tmp_dir_win%\Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /AddToPath=0 /D=%conda_parent_dir%\Miniconda3
