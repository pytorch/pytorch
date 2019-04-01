call %TMP_DIR%/ci_scripts/setup_pytorch_env.bat
cd test && python run_test.py --exclude nn --verbose && cd ..
