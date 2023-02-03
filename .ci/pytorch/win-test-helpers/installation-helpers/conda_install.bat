if "%BUILD_ENVIRONMENT%"=="" (
  set CONDA_PARENT_DIR=%CD%
) else (
  set CONDA_PARENT_DIR=C:\Jenkins
)


set tmp_dir_win=%1
set conda_parent_dir=%2

%tmp_dir_win%\Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /AddToPath=0 /D=%conda_parent_dir%\Miniconda3
