call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat
cd test && python run_test.py --exclude-jit-executor --exclude test_nn --verbose --determine-from="%1" && cd ..
if ERRORLEVEL 1 exit /b 1
