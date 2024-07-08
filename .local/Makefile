SHELL := /bin/bash

full-build:
	rm -rf build;

	USE_DISTRIBUTED=0 \
	USE_CUDA=0 \
	USE_CUDNN=0 \
	USE_CUSPARSELT=0 \
	USE_FBGEMM=0 \
	USE_KINETO=0 \
	USE_NUMPY=0 \
	USE_MKLDNN=0 \
	USE_NNPACK=0 \
	USE_QNNPACK=0 \
	USE_FLASH_ATTENTION=0 \
	USE_MEM_EFF_ATTENTION=0 \
	BUILD_TEST=0 \
	LD_LIBRARY_PATH=/opt/homebrew/opt/llvm/lib \
	python setup.py develop &> .local/build.log 


rebuild:
	USE_DISTRIBUTED=0 \
	USE_CUDA=0 \
	USE_CUDNN=0 \
	USE_CUSPARSELT=0 \
	USE_FBGEMM=0 \
	USE_KINETO=0 \
	USE_NUMPY=0 \
	USE_MKLDNN=0 \
	USE_NNPACK=0 \
	USE_QNNPACK=0 \
	USE_FLASH_ATTENTION=0 \
	USE_MEM_EFF_ATTENTION=0 \
	BUILD_TEST=0 \
	LD_LIBRARY_PATH=/opt/homebrew/opt/llvm/lib \
	python setup.py develop &> .local/build.log 