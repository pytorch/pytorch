call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat

:: Write to files (cmd has no tee); the env dump from setup must not land in uploaded artifacts.
python -u %PROJECT_DIR_WIN%\.ci\pytorch\runtime_probe.py > %TEST_DIR_WIN%\test-reports\runtime_probe.log
type %TEST_DIR_WIN%\test-reports\runtime_probe.log

python -u %PROJECT_DIR_WIN%\torch\utils\collect_env.py > %TEST_DIR_WIN%\test-reports\collect_env.log
if errorlevel 1 (
  type %TEST_DIR_WIN%\test-reports\collect_env.log
  exit /b 1
)
type %TEST_DIR_WIN%\test-reports\collect_env.log
exit /b 0
