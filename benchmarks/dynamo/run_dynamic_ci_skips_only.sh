#!/bin/bash

# This is a very simple script that runs the set of models that are currently
# skipped in dynamic shapes CI, for ease of checking if they are now working,
# and then uploads the result CSV GitHub Gist for ease of copy paste into
# Google Spreadsheet
#
# WARNING: this will silently clobber .csv and .log files in your CWD!

set -x
if getent hosts fwdproxy; then
    export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 no_proxy=.fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fburl.com,.facebook.net,.sb.fbsbx.com,localhost
fi

DATE="$(date)"
FLAG="--accuracy --training --explain --backend aot_eager --dynamic-ci-skips-only"
# shellcheck disable=SC2086 # Intended splitting of FLAG
python benchmarks/dynamo/torchbench.py --output torchbench.csv --accuracy $FLAG 2>&1 | tee torchbench.log
# shellcheck disable=SC2086 # Intended splitting of FLAG
python benchmarks/dynamo/huggingface.py --output huggingface.csv --accuracy $FLAG 2>&1 | tee huggingface.log
# shellcheck disable=SC2086 # Intended splitting of FLAG
python benchmarks/dynamo/timm_models.py --output timm_models.csv --accuracy $FLAG 2>&1 | tee timm_models.log
cat torchbench.log huggingface.log timm_models.log | tee sweep.log
gh gist create -d "Sweep logs for $(git rev-parse --abbrev-ref HEAD) $FLAG (TORCHDYNAMO_DYNAMIC_SHAPES=$TORCHDYNAMO_DYNAMIC_SHAPES) - $(git rev-parse HEAD) $DATE" sweep.log | tee -a sweep.log
python "$(dirname "$BASH_SOURCE")"/parse_logs.py sweep.log > final.csv
gh gist create final.csv
