call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat
:: exit the batch once there's an error
if not errorlevel 0 (
  echo "setup pytorch env failed"
  echo %errorlevel%
  exit /b
)

pushd test

echo Copying over test times file
copy /Y "%PYTORCH_FINAL_PACKAGE_DIR_WIN%\.pytorch-test-times.json" "%PROJECT_DIR_WIN%"

echo Run distributed tests
python distributed/test_c10d_gloo.py  -v
if ERRORLEVEL 1 goto fail

popd

:eof
exit /b 0

:fail
exit /b 1
