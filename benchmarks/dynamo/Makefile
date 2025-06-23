# Usage:
#   make build-deps BENCHMARK=<benchmark_name> MODELS=<model_names>
#
# <benchmark_name> needs to be "timm", "huggingface", or "torchbench"
# For "timm" and "huggingface", <model_names> is ignored
# For "torchbench", only models in <model_names> is installed. Support
# single model name (e.g., "alexnet"), or multiple model names (e.g.,
# "alexnet basic_gnn_gcn BERT_pytorch"), or empty (i.e., "") for installing
# all models.

clone-deps:
	# ETA: 1 min
	(cd ../../.. \
		&& (test -e torchvision || git clone --recursive https://github.com/pytorch/vision torchvision) \
		&& (test -e torchaudio || git clone --recursive https://github.com/pytorch/audio torchaudio) \
		&& (test -e torchrec || git clone --recursive https://github.com/pytorch/torchrec) \
		&& (test -e torchbenchmark || git clone --recursive https://github.com/pytorch/benchmark torchbenchmark) \
	)

pull-deps: clone-deps
	(cd ../../../torchvision    && git fetch && git checkout "$$(cat ../pytorch/.github/ci_commit_pins/vision.txt)" && git submodule update --init --recursive)
	(cd ../../../torchaudio     && git fetch && git checkout "$$(cat ../pytorch/.github/ci_commit_pins/audio.txt)" && git submodule update --init --recursive)
	(cd ../../../torchrec       && git fetch && git checkout "$$(cat ../pytorch/.github/ci_commit_pins/torchrec.txt)" && git submodule update --init --recursive)
	(cd ../../../torchbenchmark && git fetch && git checkout "$$(cat ../pytorch/.github/ci_commit_pins/torchbench.txt)" && git submodule update --init --recursive)

build-deps: clone-deps
	uv pip install astunparse numpy scipy ninja pyyaml mkl mkl-include setuptools cmake \
		typing-extensions requests protobuf numba cython scikit-learn librosa
	(cd ../../../torchvision && uv pip install -e .)
	(cd ../../../torchdata && uv pip install -e .)
	(cd ../../../torchaudio && uv pip install -e . --no-build-isolation)
	(cd ../../../FBGEMM/fbgemm_gpu && uv pip install -r requirements.txt && uv pip install -e . --no-build-isolation)
	(cd ../../../torchrec && uv pip install -e .)
	(cd ../../../detectron2 && uv pip install -e . --no-build-isolation)
	(cd ../../../torchbenchmark && python install.py --continue_on_fail)
	uv pip uninstall torchrec-nightly fbgemm-gpu-nightly
