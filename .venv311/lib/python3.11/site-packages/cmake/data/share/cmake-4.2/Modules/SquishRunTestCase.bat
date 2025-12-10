echo 'Starting the squish server...'
start %1

echo 'Running the test case...'
%2 --testcase %3 --wrapper %4 --aut %5
set result=%ERRORLEVEL%

echo 'Stopping the squish server...'
%1 --stop

exit \b %result%
