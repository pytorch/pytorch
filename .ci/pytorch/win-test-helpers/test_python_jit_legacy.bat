call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat

echo Copying over test times file
copy /Y "%PYTORCH_FINAL_PACKAGE_DIR_WIN%\.pytorch-test-times.json" "%PROJECT_DIR_WIN%"
copy /Y "%PYTORCH_FINAL_PACKAGE_DIR_WIN%\.pytorch-test-file-ratings.json" "%PROJECT_DIR_WIN%"

pushd test

echo Run jit_profiling tests
python run_test.py --include test_jit_legacy test_jit_fuser_legacy --verbose
if ERRORLEVEL 1 exit /b 1

popd
