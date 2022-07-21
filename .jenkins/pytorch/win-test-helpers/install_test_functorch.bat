call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat
:: exit the batch once there's an error
if not errorlevel 0 (
  echo "setup pytorch env failed"
  echo %errorlevel%
  exit /b
)

pushd functorch
echo "Install functorch"
python setup.py develop
if ERRORLEVEL 1 goto fail

pushd test
echo "Test functorch"
pytest test/
if ERRORLEVEL 1 goto fail
popd
popd

:eof
exit /b 0

:fail
exit /b 1
