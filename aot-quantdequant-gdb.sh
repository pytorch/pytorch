#!/bin/bash

gdb --args build/bin/aot_model_compiler \
		--model /home/ivankobzarev/nnc/quantization/quant-dequant.ptl \
		--model_name=qd \
		--model_version=v1 \
		--input_dims="1,3,224,224"

