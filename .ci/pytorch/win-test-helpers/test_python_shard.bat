call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat
setlocal enabledelayedexpansion
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

echo Copying over test times file
robocopy /E "%PYTORCH_FINAL_PACKAGE_DIR_WIN%\.additional_ci_files" "%PROJECT_DIR_WIN%\.additional_ci_files"

echo Run nn tests
if "%SHARD_NUMBER%" == "8" (
  set PYTORCH_TEST_RANGE_END=250
  python run_test.py --exclude-jit-executor --exclude-distributed-tests --include inductor/test_torchinductor_opinfo --shard "1" "1" --verbose
) else if "%SHARD_NUMBER%" == "9" (
  set PYTORCH_TEST_RANGE_START=251
  set PYTORCH_TEST_RANGE_END=500
  python run_test.py --exclude-jit-executor --exclude-distributed-tests --include inductor/test_torchinductor_opinfo --shard "1" "1" --verbose
) else if "%SHARD_NUMBER%" == "10" (
  set PYTORCH_TEST_RANGE_START=501
  python run_test.py --exclude-jit-executor --exclude-distributed-tests --include inductor/test_torchinductor_opinfo --shard "1" "1" --verbose
) else (
  set /a SHARD_COUNT=%NUM_TEST_SHARDS%-3
  python run_test.py --exclude-jit-executor --exclude-distributed-tests --exclude inductor/test_torchinductor_opinfo --shard "%SHARD_NUMBER%" "!SHARD_COUNT!" --verbose
)
if ERRORLEVEL 1 goto fail

popd

:eof
exit /b 0

:fail
exit /b 1
