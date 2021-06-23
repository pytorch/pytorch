call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat

echo Copying over test times file
copy /Y "%PYTORCH_FINAL_PACKAGE_DIR_WIN%\.pytorch-test-times.json" "%TEST_DIR_WIN%"

pushd test
python run_test.py --exclude-jit-executor --shard 1 2 --verbose --determine-from="%1"
if ERRORLEVEL 1 exit /b 1

popd
