call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat
:: exit the batch once there's an error
if not errorlevel 0 (
  echo "setup pytorch env failed"
  echo %errorlevel%
  exit /b
)

pushd test

echo Run openreg tests
python run_test.py --openreg --verbose
if ERRORLEVEL 1 goto fail

popd

:eof
exit /b 0

:fail
exit /b 1
