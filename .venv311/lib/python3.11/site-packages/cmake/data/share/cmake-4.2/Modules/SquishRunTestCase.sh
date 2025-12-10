#!/bin/sh

echo "Starting the squish server...$1 --daemon"
$1 --daemon

echo "Running the test case...$2 --testcase $3 --wrapper $4 --aut $5"
$2 --testcase $3 --wrapper $4 --aut $5
returnValue=$?

echo "Stopping the squish server...$1 --stop"
$1 --stop

exit $returnValue
