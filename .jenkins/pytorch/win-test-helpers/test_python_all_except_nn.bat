call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat
cd test && python run_test.py --exclude nn jit_simple jit_legacy jit_fuser_legacy --verbose && cd ..
if ERRORLEVEL 1 exit /b 1
