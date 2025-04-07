copy "%CUDA_PATH%\bin\cusparse*64_*.dll*" pytorch\torch\lib
copy "%CUDA_PATH%\bin\cublas*64_*.dll*" pytorch\torch\lib
copy "%CUDA_PATH%\bin\cudart*64_*.dll*" pytorch\torch\lib
copy "%CUDA_PATH%\bin\curand*64_*.dll*" pytorch\torch\lib
copy "%CUDA_PATH%\bin\cufft*64_*.dll*" pytorch\torch\lib
copy "%CUDA_PATH%\bin\cusolver*64_*.dll*" pytorch\torch\lib

copy "%CUDA_PATH%\bin\cudnn*64_*.dll*" pytorch\torch\lib
copy "%CUDA_PATH%\bin\nvrtc*64_*.dll*" pytorch\torch\lib
copy "%CUDA_PATH%\extras\CUPTI\lib64\cupti64_*.dll*" pytorch\torch\lib

copy "C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64\nvToolsExt64_1.dll*" pytorch\torch\lib
copy "%CONDA_LIB_PATH%\libiomp*5md.dll" pytorch\torch\lib

:: Should be set in build_pytorch.bat
copy "%libuv_ROOT%\bin\uv.dll" pytorch\torch\lib

::copy zlib if it exist in windows/system32
if exist "C:\Windows\System32\zlibwapi.dll" (
    copy "C:\Windows\System32\zlibwapi.dll"  pytorch\torch\lib
)

::copy nvJitLink dll is requires for cuda 12+
if exist "%CUDA_PATH%\bin\nvJitLink_*.dll*" (
    copy "%CUDA_PATH%\bin\nvJitLink_*.dll*"  pytorch\torch\lib
)
