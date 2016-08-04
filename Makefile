# Add main target here - setup.py doesn't understand the need to recompile
# after generic files change
.PHONY: all clean torch

all: install

torch:
	python3 setup.py build

install:
	python3 setup.py install

clean:
	@rm -rf build
	@rm -rf dist
	@rm -rf torch.egg-info
	@rm -rf tools/__pycache__
	@rm -rf torch/csrc/generic/TensorMethods.cpp
	@rm -rf torch/lib/tmp_install
	@rm -rf torch/lib/build
	@rm -rf torch/lib/*.so
	@rm -rf torch/lib/*.h
