call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat

cd test\custom_operator

:: Build the custom operator library.
mkdir build
cd build
:: Note: Caffe2 does not support MSVC + CUDA + Debug mode (has to be Release mode)
cmake -DCMAKE_PREFIX_PATH=%TMP_DIR_WIN%\build\torch -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja -v
cd ..

:: Run tests Python-side and export a script module.
python test_custom_ops.py -v
python model.py --export-script-module="build/model.pt"
:: Run tests C++-side and load the exported script module.
cd build
set PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64;%TMP_DIR_WIN%\build\torch\lib;%PATH%
test_custom_ops.exe model.pt
