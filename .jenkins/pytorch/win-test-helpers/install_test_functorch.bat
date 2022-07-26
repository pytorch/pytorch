call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat
:: exit the batch once there's an error
if not errorlevel 0 (
  echo "setup pytorch env failed"
  echo %errorlevel%
  exit /b
)

pushd functorch
echo "Install functorch"
:: --no-deps because for some reason, on windows, `torch` isn't found in
:: `pip list` despite being installed. With just `python setup.py develop`,
:: setuptools explicitly checks for the existence of torch and can't find it.
python setup.py develop --no-deps
popd
if ERRORLEVEL 1 goto fail

echo "Installing test dependencies"
pip install networkx
if errorlevel 1 exit /b

echo "Test functorch"
pushd test
python run_test.py --functorch --shard "%SHARD_NUMBER%" "%NUM_TEST_SHARDS%" --verbose
popd
if ERRORLEVEL 1 goto fail

:eof
exit /b 0

:fail
exit /b 1
