call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat
:: exit the batch once there's an error
if not errorlevel 0 (
    echo "setup pytorch env failed"
    echo %errorlevel%
    exit /b
)

pushd test
python run_test.py --exclude-jit-executor --verbose
popd
if ERRORLEVEL 1 exit /b 1
