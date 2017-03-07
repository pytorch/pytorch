:: Installation scripts for appveyor.

@echo Downloading CUDA toolkit 8 ...

appveyor DownloadFile ^
  https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_windows-exe ^
  -FileName cuda_8.0.44_windows.exe
appveyor Downloadfile ^
  http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-windows10-x64-v5.1.zip ^
  -FileName cudnn-8.0-windows10-x64-v5.1.zip

@echo Installing CUDA toolkit 8 ...
cuda_8.0.44_windows.exe -s compiler_8.0 cublas_8.0 cublas_dev_8.0 cudart_8.0 curand_8.0 curand_dev_8.0 nvrtc_8.0 nvrtc_dev_8.0
set PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin;%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\libnvvp;%PATH%
:: TODO: we will still need to figure out how to install cudnn.
7z x cudnn-8.0-windows10-x64-v5.1.zip
copy cuda\include\cudnn.h ^
  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include\"
copy cuda\lib\x64\cudnn.lib ^
  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64\"
copy cuda\bin\cudnn64_5.dll ^
  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\"

:: Make sure that nvcc is working correctly.
nvcc -V || exit /b

:: Miniconda path for appveyor
set PATH=C:\Miniconda-x64;C:\Miniconda-x64\Scripts;%PATH%
:: Install numpy
conda install -y numpy