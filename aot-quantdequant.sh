#!/bin/bash

build/bin/aot_model_compiler \
		--model /home/ivankobzarev/nnc/quantization/quant-dequant.ptl \
		--model_name=qd \
		--model_version=v1 \
		--input_dims="1,3,224,224"

build/bin/aot_model_compiler \
		--model /home/ivankobzarev/nnc/quantization/quant-dequant-3arg.ptl \
		--model_name=qd3arg \
		--model_version=v1 \
		--input_dims="1,3,224,224"
