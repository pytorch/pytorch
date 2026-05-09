#!/bin/bash -xe
#
# Default to on, unless set to off by the calling script.
VALGRIND=${VALGRIND:=ON}
export CPP_TESTS_DIR=$1
if [ "$VALGRIND" == "ON" ]; then
    # --leak-check=full is necessary to return 1 in case of definite and possible leaks.
    valgrind   --error-exitcode=1  --leak-check=full ${CPP_TESTS_DIR}/test_shim
fi
