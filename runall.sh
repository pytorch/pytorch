#!/bin/bash
set -x
if getent hosts fwdproxy; then
    export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 no_proxy=.fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fburl.com,.facebook.net,.sb.fbsbx.com,localhost
fi
export TORCHDYNAMO_DYNAMIC_SHAPES=1
export AOT_DYNAMIC_SHAPES=1
DATE="$(date)"
#FLAG="--backend inductor"
FLAG="--accuracy --backend ${BACKEND:-aot_eager} --training"
#FLAG="--accuracy --backend inductor --training"
#FLAG="--accuracy --backend inductor"
# shellcheck disable=SC2086 # Intended splitting of FLAG
python benchmarks/dynamo/torchbench.py --output torchbench.csv --accuracy $FLAG 2>&1 | tee torchbench.log
# shellcheck disable=SC2086 # Intended splitting of FLAG
python benchmarks/dynamo/huggingface.py --output huggingface.csv --accuracy $FLAG 2>&1 | tee huggingface.log
# shellcheck disable=SC2086 # Intended splitting of FLAG
python benchmarks/dynamo/timm_models.py  --output timm_models.csv --accuracy $FLAG 2>&1 | tee timm_models.log
cat torchbench.log huggingface.log timm_models.log |
    tee sweep.log |
    gh gist create -d "Sweep logs for $(git rev-parse --abbrev-ref HEAD) $FLAG (TORCHDYNAMO_DYNAMIC_SHAPES=$TORCHDYNAMO_DYNAMIC_SHAPES) - $(git rev-parse HEAD) $DATE" - |
    tee url.log
python log_extract.py sweep.log > final.csv
gh gist create final.csv
