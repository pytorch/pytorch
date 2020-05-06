call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat
if errorlevel 1 exit /b 1

cd %TMP_DIR_WIN%\build\torch\bin
set TEST_OUT_DIR=%~dp0\..\..\..\test\test-reports\cpp-unittest
md %TEST_OUT_DIR%
set PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64;%TMP_DIR_WIN%\build\torch\lib;%PATH%

test_api.exe --gtest_filter="-IntegrationTest.MNIST*" --gtest_output=xml:%TEST_OUT_DIR%\test_api.xml
if errorlevel 1 exit /b 1
if not errorlevel 0 exit /b 1

cd %TMP_DIR_WIN%\build\torch\test
for /r "." %%a in (*.exe) do (
    call :libtorch_check "%%~na" "%%~fa"
    if errorlevel 1 exit /b 1
)

goto :eof

:libtorch_check
set TEST_NAME=%~1
set TEST_BINARY=%~2
:: Skip verify_api_visibility as it a compile level test
if "%TEST_NAME%" == "verify_api_visibility" goto :eof

:: See https://github.com/pytorch/pytorch/issues/25161
if "%TEST_NAME%" == "c10_metaprogramming_test" goto :eof
if "%TEST_NAME%" == "module_test" goto :eof
:: See https://github.com/pytorch/pytorch/issues/25312
if "%TEST_NAME%" == "converter_nomigraph_test" goto :eof
:: See https://github.com/pytorch/pytorch/issues/35636
if "%TEST_NAME%" == "generate_proposals_op_gpu_test" goto :eof
:: See https://github.com/pytorch/pytorch/issues/35648
if "%TEST_NAME%" == "reshape_op_gpu_test" goto :eof
:: See https://github.com/pytorch/pytorch/issues/35651
if "%TEST_NAME%" == "utility_ops_gpu_test" goto :eof

:: Skip cuda tests on a cpu agent
if "%USE_CUDA%" == "0" (
  if NOT "%TEST_NAME:cuda=%" == "%TEST_NAME%" goto :eof
  if NOT "%TEST_NAME:gpu=%" == "%TEST_NAME%" goto :eof
  if NOT "%TEST_NAME:cudnn=%" == "%TEST_NAME%" goto :eof
)

echo Running "%TEST_BINARY%"
if "%TEST_NAME%" == "c10_intrusive_ptr_benchmark" (
  call "%TEST_BINARY%"
  goto :eof
)
call "%TEST_BINARY%" --gtest_output=xml:%TEST_OUT_DIR%\%TEST_NAME%.xml
if errorlevel 1 (
  echo %TEST_NAME% failed with exit code %errorlevel%
  exit /b 1
)
if not errorlevel 0 (
  echo %TEST_NAME% failed with exit code %errorlevel%
  exit /b 1
)

goto :eof
