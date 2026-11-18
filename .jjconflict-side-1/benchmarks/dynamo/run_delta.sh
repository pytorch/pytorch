#!/bin/bash

set -x

# Some QoL for people running this script on Meta servers
if getent hosts fwdproxy; then
    export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 no_proxy=.fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fburl.com,.facebook.net,.sb.fbsbx.com,localhost
fi

WORK="$PWD"

cd "$(dirname "$BASH_SOURCE")"/../..

ROOT="$PWD"

mkdir -p "$WORK/sweep/static"
mkdir -p "$WORK/sweep/dynamic"

(cd "$WORK/sweep/static" && "$ROOT/benchmarks/dynamo/run_all.sh" "$@")
(cd "$WORK/sweep/dynamic" && "$ROOT/benchmarks/dynamo/run_all.sh" "$@" --dynamic-shapes)
python benchmarks/dynamo/combine_csv.py "$WORK/sweep/static/final.csv" "$WORK/sweep/dynamic/final.csv" > "$WORK/delta.csv"
gh gist create "$WORK/delta.csv"
