#!/bin/bash

# This is a very simple script that runs all three benchmark suites
# and then runs a log parsing script to get the logs into a csv form
# and uploads the result to GitHub Gist for easy copy paste into Google
# Spreadsheet.
#
# Useful flag sets:
#
# Run models that are skipped in dynamic CI only, to update skips
#   ./run_all.sh --training --backend aot_eager --dynamic-ci-skips-only
#
# Run CI models with dynamic shapes, training, aot_eager
#   ./run_all.sh --training --backend aot_eager --dynamic-shapes --ci
#
# Run CI models with dynamic shapes, inference, inductor
#   ./run_all.sh --backend inductor --dynamic-shapes --ci
#
# WARNING: this will silently clobber .csv and .log files in your CWD!

set -x

# Some QoL for people running this script on Meta servers
if getent hosts fwdproxy; then
    export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 no_proxy=.fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fburl.com,.facebook.net,.sb.fbsbx.com,localhost
fi

# Feel free to edit these, but we expect most users not to need to modify this
BASE_FLAGS=( --accuracy --explain --timing --print-graph-breaks )
DATE="$(date)"
WORK="$PWD"

cd "$(dirname "$BASH_SOURCE")"/../..

python benchmarks/dynamo/benchmarks.py --output "$WORK"/benchmarks.csv "${BASE_FLAGS[@]}" "$@" 2>&1 | tee "$WORK"/sweep.log
gh gist create -d "Sweep logs for $(git rev-parse --abbrev-ref HEAD) $* - $(git rev-parse HEAD) $DATE" "$WORK"/sweep.log | tee -a "$WORK"/sweep.log
python benchmarks/dynamo/parse_logs.py "$WORK"/sweep.log > "$WORK"/final.csv
gh gist create "$WORK"/final.csv
