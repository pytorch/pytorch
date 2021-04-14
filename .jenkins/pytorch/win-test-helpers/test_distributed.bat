REM The first argument should lead to the python interpreter
%1\python.exe test/run_test.py --verbose -i distributed/test_jit_c10d
if %errorlevel% neq 0 ( exit /b %errorlevel% )

%1\python.exe test/run_test.py --verbose -i distributed/test_c10d
if %errorlevel% neq 0 ( exit /b %errorlevel% )

%1\python test/run_test.py --verbose -i distributed/test_c10d_spawn
if %errorlevel% neq 0 ( exit /b %errorlevel% )

%1\python.exe test/run_test.py --verbose -i distributed/test_data_parallel
if %errorlevel% neq 0 ( exit /b %errorlevel% )
