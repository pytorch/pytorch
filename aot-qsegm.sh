#!/bin/bash
ls qsegm.ptl || jf download GJg_NQzh9A4v3xwDALcNy0nhLD8ybsIXAAAz --file "qsegm.ptl"

PYTORCH_JIT_LOG_LEVEL=">>kernel:>>eval:>>symbolic_shape_analysis" \
build/bin/aot_model_compiler \
    --model qsegm.ptl \
    --model_name=mnetv3-large \
    --model_version=v1 \
    --input_dims="1,4,224,224" && \
llc qsegm.compiled.ll -march=x86-64
