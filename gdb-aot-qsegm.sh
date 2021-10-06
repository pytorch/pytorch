#!/bin/bash
gdb --args build/bin/aot_model_compiler \
    --model segm.pt \
    --model_name=mnetv3-large \
    --model_version=v1 \
    --input_dims="1,4,224,224"
