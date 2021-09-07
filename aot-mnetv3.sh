#!/bin/bash

build/bin/aot_model_compiler \
    --model /home/ivankobzarev/nnc/mobilenetv3.pt \
		--model_name=mnetv3-large \
		--model_version=v1 \
		--input_dims="1,3,224,224"
