:: Installation scripts for appveyor.

@echo on

if "%USE_CUDA%" == "ON" call %~dp0%install_cuda.bat

:: Miniconda path for appveyor
set PATH=C:\Miniconda-x64;C:\Miniconda-x64\Scripts;%PATH%
:: Install numpy
conda install -y numpy
