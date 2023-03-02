:: Skip LibTorch tests when building a GPU binary and testing on a CPU machine
:: because LibTorch tests are not well designed for this use case.
if "%USE_CUDA%" == "0" IF NOT "%CUDA_VERSION%" == "cpu" exit /b 0

call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat
if errorlevel 1 exit /b 1

cd %TMP_DIR_WIN%\build\torch\bin
set TEST_OUT_DIR=%~dp0\..\..\..\test\test-reports\cpp-unittest
md %TEST_OUT_DIR%
set PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64;%TMP_DIR_WIN%\build\torch\lib;%PATH%

set TEST_API_OUT_DIR=%TEST_OUT_DIR%\test_api
md %TEST_API_OUT_DIR%
test_api.exe --gtest_filter="-IntegrationTest.MNIST*" --gtest_output=xml:%TEST_API_OUT_DIR%\test_api.xml
if errorlevel 1 exit /b 1
if not errorlevel 0 exit /b 1

cd %TMP_DIR_WIN%\build\torch\test
for /r "." %%a in (*.exe) do (
    call :libtorch_check "%%~na" "%%~fa"
    if errorlevel 1 exit /b 1
)

goto :eof

:libtorch_check
:: Skip verify_api_visibility as it a compile level test
if "%~1" == "verify_api_visibility" goto :eof

:: See https://github.com/pytorch/pytorch/issues/25161
if "%~1" == "c10_metaprogramming_test" goto :eof
if "%~1" == "module_test" goto :eof
:: See https://github.com/pytorch/pytorch/issues/25312
if "%~1" == "converter_nomigraph_test" goto :eof
:: See https://github.com/pytorch/pytorch/issues/35636
if "%~1" == "generate_proposals_op_gpu_test" goto :eof
:: See https://github.com/pytorch/pytorch/issues/35648
if "%~1" == "reshape_op_gpu_test" goto :eof
:: See https://github.com/pytorch/pytorch/issues/35651
if "%~1" == "utility_ops_gpu_test" goto :eof

echo Running "%~2"
if "%~1" == "c10_intrusive_ptr_benchmark" (
  call "%~2"
  goto :eof
)
:: Differentiating the test report directories is crucial for test time reporting.
md %TEST_OUT_DIR%\%~n2
call "%~2" --gtest_output=xml:%TEST_OUT_DIR%\%~n2\%~1.xml
if errorlevel 1 (
  echo %1 failed with exit code %errorlevel%
  exit /b 1
)
if not errorlevel 0 (
  echo %1 failed with exit code %errorlevel%
  exit /b 1
)

goto :eof
