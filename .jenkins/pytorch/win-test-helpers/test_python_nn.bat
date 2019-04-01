call %TMP_DIR%/ci_scripts/setup_pytorch_env.bat
:: Some smoke tests
cd test
:: Checking that caffe2.python is available
python -c "from caffe2.python import core"
if ERRORLEVEL 1 exit /b 1
:: Checking that MKL is available
python -c "import torch; exit(0 if torch.backends.mkl.is_available() else 1)"
if ERRORLEVEL 1 exit /b 1
:: Checking that CUDA archs are setup correctly
python -c "import torch; torch.randn([3,5]).cuda()"
if ERRORLEVEL 1 exit /b 1
:: Checking that magma is available
python -c "import torch; torch.rand(1).cuda(); exit(0 if torch.cuda.has_magma else 1)"
if ERRORLEVEL 1 exit /b 1
:: Checking that CuDNN is available
python -c "import torch; exit(0 if torch.backends.cudnn.is_available() else 1)"
if ERRORLEVEL 1 exit /b 1
cd ..

:: Run nn tests
cd test/ && python run_test.py --include nn --verbose && cd ..
