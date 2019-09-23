call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat

cd %TMP_DIR_WIN%\build\torch\bin
set PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64;%TMP_DIR_WIN%\build\torch\lib;%PATH%
test_api.exe --gtest_filter="-IntegrationTest.MNIST*"

if errorlevel 1 exit /b 1

cd %TMP_DIR_WIN%\build\torch\test
for /r "." %%a in (*.exe) do (
    call :libtorch_check "%%~na" "%%~fa"
)

goto :eof

:libtorch_check
:: See https://github.com/pytorch/pytorch/issues/25161
if "%~1" == "c10_metaprogramming_test" goto :eof
if "%~1" == "module_test" goto :eof
:: See https://github.com/pytorch/pytorch/issues/25312
if "%~1" == "converter_nomigraph_test" goto :eof

echo Running "%~2"
call "%~2"
if errorlevel 1 exit /b 1

goto :eof
