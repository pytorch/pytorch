call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat
if errorlevel 1 exit /b 1

:: Save the current working directory so that we can go back there
set CWD=%cd%

set CPP_TESTS_DIR=%TMP_DIR_WIN%\build\torch\bin
set PATH=%TMP_DIR_WIN%\build\torch\lib;%PATH%

set TORCH_CPP_TEST_MNIST_PATH=%CWD%\test\cpp\api\mnist
python tools\download_mnist.py --quiet -d %TORCH_CPP_TEST_MNIST_PATH%

python test\run_test.py --cpp --verbose -i cpp/test_api
if errorlevel 1 exit /b 1
if not errorlevel 0 exit /b 1

cd %TMP_DIR_WIN%\build\torch\test

:: Enable delayed variable expansion to make the list
setlocal enabledelayedexpansion
set EXE_LIST=
for /r "." %%a in (*.exe) do (
  set EXE_LIST=!EXE_LIST! cpp/%%~fa
)

:: Run python test\run_test.py on the list
python test\run_test.py --cpp --verbose -i !EXE_LIST! ^
  --exclude ^
  :: Skip verify_api_visibility as it a compile level test
  "cpp/verify_api_visibility" ^
  :: NB: This is not a gtest executable file, thus couldn't be handled by pytest-cpp
  "cpp/c10_intrusive_ptr_benchmark"
if errorlevel 1 goto fail
if not errorlevel 0 goto fail

goto :eof

:eof
exit /b 0

:fail
exit /b 1
