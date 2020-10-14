call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat

@echo on
pushd test

echo Some smoke tests
"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\gflags.exe" /i python.exe +sls
python %SCRIPT_HELPERS_DIR%\run_python_nn_smoketests.py
if ERRORLEVEL 1 exit /b 1

"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\gflags.exe" /i python.exe -sls
if ERRORLEVEL 1 exit /b 1

echo Run caffe2 ops tests
pip list
pip install pytest
python -m pytest --maxfail=100 -x -v --disable-warnings --junit-xml="resulst.xml" %PYTHONPATH%\caffe2\python\operator_test -G

if ERRORLEVEL 1 exit /b 1

popd



