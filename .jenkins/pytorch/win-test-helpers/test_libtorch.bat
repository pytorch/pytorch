call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat

cd %TMP_DIR_WIN%\build\torch\bin
set PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64;%TMP_DIR_WIN%\build\torch\lib;%PATH%
test_api.exe --gtest_filter="-IntegrationTest.MNIST*"

if errorlevel 1 exit /b 1

cd %TMP_DIR_WIN%\build\torch\test
for /r "." %%a in (*.exe) do (
    echo Running "%%~fa"
    if "%%~na" == "c10_Metaprogramming_test" (
        echo Skipping "%%~fa" because it is broken
    ) else (
        if "%%~na" == "module_test" (
            echo Skipping "%%~fa" because it is broken
        ) else (
            call "%%~fa"
            if errorlevel 1 exit /b 1
        )
    )
)
