call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat
:: exit the batch once there's an error
if not errorlevel 0 (
  echo "setup pytorch env failed"
  echo %errorlevel%
  exit /b
)

pushd test

set GFLAGS_EXE="C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\gflags.exe"
if "%SHARD_NUMBER%" == "1" (
  if exist %GFLAGS_EXE% (
    echo Some smoke tests
    %GFLAGS_EXE% /i python.exe +sls
    python %SCRIPT_HELPERS_DIR%\run_python_nn_smoketests.py
    if ERRORLEVEL 1 goto fail

    %GFLAGS_EXE% /i python.exe -sls
    if ERRORLEVEL 1 goto fail
  )
)

echo Run nn tests
python run_test.py --exclude-jit-executor --exclude-distributed-tests --shard "%SHARD_NUMBER%" "%NUM_TEST_SHARDS%" --verbose
if ERRORLEVEL 1 goto fail

popd

:eof
exit /b 0

:fail
exit /b 1
