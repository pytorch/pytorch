call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat

pushd test

echo Run jit_profiling tests
python run_test.py --include test_jit_profiling test_jit_fuser_te test_tensorexpr --verbose --determine-from="%1"
if ERRORLEVEL 1 exit /b 1

popd


