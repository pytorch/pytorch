call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat

set GFLAGS_EXE="C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\gflags.exe"
if exist %GFLAGS_EXE% (
    echo Some smoke tests
    %GFLAGS_EXE% /i python.exe +sls
    python %SCRIPT_HELPERS_DIR%\run_python_nn_smoketests.py
    if ERRORLEVEL 1 exit /b 1

    %GFLAGS_EXE% /i python.exe -sls
    if ERRORLEVEL 1 exit /b 1
)

echo Copying over test times file
copy /Y "%PYTORCH_FINAL_PACKAGE_DIR_WIN%\.pytorch-test-times.json" "%TEST_DIR_WIN%"

echo Run nn tests
pushd test

if "%RUN_SMOKE_TESTS_ONLY%"=="1" (
    :: Download specified test cases to run
    curl --retry 3 -k https://raw.githubusercontent.com/janeyx99/test-infra/add-windows-smoke-tests/stats/windows_smoke_tests.csv --output .pytorch_specified_test_cases.csv
    if ERRORLEVEL 1 exit /b 1

    python run_test.py --exclude-jit-executor --shard 1 2 --verbose --determine-from="%1" --run-specified-test-cases
) else (
    python run_test.py --exclude-jit-executor --shard 1 2 --verbose --determine-from="%1"
)
if ERRORLEVEL 1 exit /b 1

popd
