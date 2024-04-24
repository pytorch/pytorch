#!/bin/bash

# Typical usage:
#
#   scripts/analysis/run_test_csv.sh test/inductor/test_torchinductor.py

set -x

if getent hosts fwdproxy; then
    export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 no_proxy=.fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fburl.com,.facebook.net,.sb.fbsbx.com,localhost
fi
TEST_FILE="$1"
TEST_ARGS="$*"  # includes file name
shift
pytest --csv "$TEST_FILE.csv" -v "$TEST_FILE" "$@" 2>&1 | tee "$TEST_FILE.log"
LOG_URL="$(gh gist create -d "Test logs for $TEST_ARGS" "$TEST_FILE.log")"
python "$(dirname "$BASH_SOURCE")"/format_test_csv.py --log-url "$LOG_URL" "$TEST_FILE.csv" | gh gist create -
