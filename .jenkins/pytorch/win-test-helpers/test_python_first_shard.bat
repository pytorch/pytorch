call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat


pushd test

echo Some smoke tests
"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\gflags.exe" /i python.exe +sls
python %SCRIPT_HELPERS_DIR%\run_python_nn_smoketests.py
if ERRORLEVEL 1 exit /b 1

"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\gflags.exe" /i python.exe -sls
if ERRORLEVEL 1 exit /b 1

echo Run nn tests
python run_test.py --exclude-jit-executor --shard 1 2 --verbose --determine-from="%1"
if ERRORLEVEL 1 exit /b 1

popd
