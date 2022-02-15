call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat
:: exit the batch once there's an error
if not errorlevel 0 (
    echo "setup pytorch env failed"
    echo %errorlevel%
    exit /b
)

pushd test
if "%RUN_SMOKE_TESTS_ONLY%"=="1" (
    :: Download specified test cases to run
    curl --retry 3 -k https://raw.githubusercontent.com/pytorch/test-infra/main/stats/windows_smoke_tests.csv --output .pytorch_specified_test_cases.csv
    if ERRORLEVEL 1 exit /b 1

    python run_test.py --exclude-jit-executor --verbose --run-specified-test-cases
) else (
    python run_test.py --exclude-jit-executor --verbose
)
popd
if ERRORLEVEL 1 exit /b 1
