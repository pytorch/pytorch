call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat


pushd test

echo %cd%
python -c "import os; import torch; print(os.path.realpath(torch.__file__))"
python -c "import os; import caffe2; print(os.path.realpath(caffe2.__file__))"
python -c "import os; import caffe2; print(os.path.dirname(os.path.realpath(caffe2.__file__)))"

echo %python_installation%

echo Some smoke tests
"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\gflags.exe" /i python.exe +sls
python %SCRIPT_HELPERS_DIR%\run_python_nn_smoketests.py
if ERRORLEVEL 1 exit /b 1

"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\gflags.exe" /i python.exe -sls
if ERRORLEVEL 1 exit /b 1

echo Run caffe2 ops tests
pip list
pip install pytest
rem for /F "usebackq" %f IN (`python -c "import os; import caffe2; print(os.path.dirname(os.path.realpath(caffe2.__file__)))"`) DO set caffe2_dir=%f
python -m pytest -x -v --disable-warnings --junit-xml="resulst.xml" C:\Users\circleci\project\build\win_tmp\build\caffe2\python\operator_test -G

rem pytorch_installation="$(dirname $(dirname $(cd $TMP_DIR && python -c 'import os; import torch; print(os.path.realpath(torch.__file__))')))"
rem python_installation="$(dirname $(dirname $(cd $TMP_DIR && python -c 'import os; import caffe2; print(os.path.realpath(caffe2.__file__))')))"
rem caffe2_pypath="$python_installation/caffe2"
rem python run_test.py --include test_nn --verbose --determine-from="%1"
if ERRORLEVEL 1 exit /b 1

popd


