call %TMP_DIR%/ci_scripts/setup_pytorch_env.bat
dir
dir %TMP_DIR_WIN%\build
dir %TMP_DIR_WIN%\build\torch
dir %TMP_DIR_WIN%\build\torch\lib
cd %TMP_DIR_WIN%\build\torch\lib
set PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64;%TMP_DIR_WIN%\build\torch\lib;%PATH%
test_api.exe --gtest_filter="-IntegrationTest.MNIST*"
