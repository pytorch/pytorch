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
	sh setup.sh $(BENCHMARK) "$(MODELS)"
