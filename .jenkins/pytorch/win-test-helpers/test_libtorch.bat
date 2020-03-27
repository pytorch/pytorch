call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat

cd %TMP_DIR_WIN%\build\torch\bin
set TEST_OUT_DIR=%~dp0\..\..\..\test\test-reports\cpp-unittest
md %TEST_OUT_DIR%
set PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64;%TMP_DIR_WIN%\build\torch\lib;%PATH%
test_api.exe --gtest_filter="-IntegrationTest.MNIST*" --gtest_output=xml:%TEST_OUT_DIR%\test_api.xml

if not errorlevel 0 exit /b 1

cd %TMP_DIR_WIN%\build\torch\test
for /r "." %%a in (*.exe) do (
    call :libtorch_check "%%~na" "%%~fa"
    if errorlevel 1 exit /b 1
)

goto :eof

:libtorch_check
:: See https://github.com/pytorch/pytorch/issues/25161
if "%~1" == "c10_metaprogramming_test" goto :eof
if "%~1" == "module_test" goto :eof
:: See https://github.com/pytorch/pytorch/issues/25312
if "%~1" == "converter_nomigraph_test" goto :eof

echo Running "%~2"
call "%~2" --gtest_output=xml:%TEST_OUT_DIR%\%~1.xml
if not errorlevel 0 exit /b 1

goto :eof
