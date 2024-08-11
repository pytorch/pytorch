call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat

echo Copying over test times file
robocopy /E "%PYTORCH_FINAL_PACKAGE_DIR_WIN%\.additional_ci_files" "%PROJECT_DIR_WIN%\.additional_ci_files"

pushd test

echo Run torch inductor tests
python run_test.py --include inductor/test_cpu_cpp_wrapper inductor/test_torchinductor --verbose
if ERRORLEVEL 1 exit /b 1

popd
