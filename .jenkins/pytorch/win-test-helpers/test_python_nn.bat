call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat


pushd test

echo %cd%
python -c 'import os; import torch; print(os.path.realpath(torch.__file__))'
python -c 'import os; import caffe2; print(os.path.realpath(caffe2.__file__))'
python -m pip show torch
echo %python_installation%

echo Some smoke tests
"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\gflags.exe" /i python.exe +sls
python %SCRIPT_HELPERS_DIR%\run_python_nn_smoketests.py
if ERRORLEVEL 1 exit /b 1

"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\gflags.exe" /i python.exe -sls
if ERRORLEVEL 1 exit /b 1

echo Run nn tests
python run_test.py --include test_nn --verbose --determine-from="%1"
if ERRORLEVEL 1 exit /b 1

popd


