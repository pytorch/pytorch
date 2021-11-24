#!/bin/bash
PYTORCH_JIT_LOG_LEVEL=">>kernel:>>shape_analysis:>>symbolic_shape_analysis:>>symbolic_shape_runtime_fusion:>>symbolic_shape_registry" \
gdb --args build/bin/aot_model_compiler \
    --model /home/ivankobzarev/nnc/quantization/bi_bytedoc/qbi.ptl \
    --model_name=qbi \
    --model_version=v1 \
    --input_dims="1,115"
