call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat

echo Copying over test times file

pushd test

echo Run jit_profiling tests
python run_test.py --include test_jit_legacy test_jit_fuser_legacy --verbose
if ERRORLEVEL 1 exit /b 1

popd
