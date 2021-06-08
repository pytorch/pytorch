call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat

echo Copying over test times file
copy /Y "%PYTORCH_FINAL_PACKAGE_DIR_WIN%\.pytorch-test-times" "%TEST_DIR_WIN%"

cd test && python run_test.py --exclude-jit-executor --shard 2 2 --verbose --determine-from="%1" && cd ..

if ERRORLEVEL 1 exit /b 1
