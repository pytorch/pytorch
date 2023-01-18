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
if getent hosts fwdproxy; then
    export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 no_proxy=.fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fburl.com,.facebook.net,.sb.fbsbx.com,localhost
fi

# Feel free to edit these, but we expect most users not to need to modify this
BASE_FLAGS=( --accuracy --explain )
DATE="$(date)"

python benchmarks/dynamo/torchbench.py --output torchbench.csv "${BASE_FLAGS[@]}" "$@" 2>&1 | tee torchbench.log
python benchmarks/dynamo/huggingface.py --output huggingface.csv "${BASE_FLAGS[@]}" "$@" 2>&1 | tee huggingface.log
python benchmarks/dynamo/timm_models.py --output timm_models.csv "${BASE_FLAGS[@]}" "$@" 2>&1 | tee timm_models.log
cat torchbench.log huggingface.log timm_models.log | tee sweep.log
gh gist create -d "Sweep logs for $(git rev-parse --abbrev-ref HEAD) $@ - $(git rev-parse HEAD) $DATE" sweep.log | tee -a sweep.log
python "$(dirname "$BASH_SOURCE")"/parse_logs.py sweep.log > final.csv
gh gist create final.csv
