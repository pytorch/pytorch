set SQUISHSERVER=%1
set SQUISHRUNNER=%2
set TESTSUITE=%3
set TESTCASE=%4
set AUT=%5
set AUTDIR=%6

%SQUISHSERVER% --stop

echo "Adding AUT... %SQUISHSERVER% --config addAUT %AUT% %AUTDIR%"
%SQUISHSERVER% --config addAUT "%AUT%" "%AUTDIR%"

echo "Starting the squish server... %SQUISHSERVER%"
start /B "Squish Server" %SQUISHSERVER%

echo "Running the test case... %SQUISHRUNNER% --testsuite %TESTSUITE% --testcase %TESTCASE%"
%SQUISHRUNNER% --testsuite "%TESTSUITE%" --testcase "%TESTCASE%"
set returnValue=%ERRORLEVEL%

echo "Stopping the squish server... %SQUISHSERVER% --stop"
%SQUISHSERVER% --stop

exit /B %returnValue%
