# Usage:
#   make build-deps TORCHBENCH_MODELS=<model_names>
#   Support install a single torchbench model (e.g., "alexnet"),
#   or multiple torchbench model names (e.g., "alexnet basic_gnn_gcn BERT_pytorch"),
#   or empty (i.e., "") for installing all torchbench models.

clone-deps:
	(cd ../../.. \
		&& (test -e torchvision || git clone --recursive https://github.com/pytorch/vision torchvision) \
		&& (test -e torchdata || git clone --recursive https://github.com/pytorch/data.git torchdata) \
		&& (test -e torchtext || git clone --recursive https://github.com/pytorch/text torchtext) \
		&& (test -e torchaudio || git clone --recursive https://github.com/pytorch/audio torchaudio) \
		&& (test -e detectron2 || git clone --recursive https://github.com/facebookresearch/detectron2) \
		&& (test -e FBGEMM || git clone --recursive https://github.com/pytorch/FBGEMM) \
		&& (test -e torchrec || git clone --recursive https://github.com/pytorch/torchrec) \
		&& (test -e torchbenchmark || git clone --recursive https://github.com/pytorch/benchmark torchbenchmark) \
	)

pull-deps: clone-deps
	(cd ../../../torchvision    && git fetch && git checkout "$$(cat ../pytorch/.github/ci_commit_pins/vision.txt)" && git submodule update --init --recursive)
	(cd ../../../torchdata      && git fetch && git checkout "$$(cat ../pytorch/.github/ci_commit_pins/data.txt)" && git submodule update --init --recursive)
	(cd ../../../torchtext      && git fetch && git checkout "$$(cat ../pytorch/.github/ci_commit_pins/text.txt)" && git submodule update --init --recursive)
	(cd ../../../torchaudio     && git fetch && git checkout "$$(cat ../pytorch/.github/ci_commit_pins/audio.txt)" && git submodule update --init --recursive)
	(cd ../../../FBGEMM         && git fetch && git checkout "$$(cat ../pytorch/.github/ci_commit_pins/fbgemm.txt)" && git submodule update --init --recursive)
	(cd ../../../torchrec       && git fetch && git checkout "$$(cat ../pytorch/.github/ci_commit_pins/torchrec.txt)" && git submodule update --init --recursive)
	(cd ../../../detectron2     && git fetch && git checkout HEAD && git submodule update --init --recursive)
	(cd ../../../torchbenchmark && git fetch && git checkout "$$(cat ../pytorch/.github/ci_commit_pins/torchbench.txt)" && git submodule update --init --recursive)

build-deps: clone-deps
	uv pip install astunparse numpy scipy ninja pyyaml mkl mkl-include setuptools cmake \
		typing-extensions requests protobuf numba cython scikit-learn librosa
	(cd ../../../torchvision && uv pip install -e . --no-build-isolation)
	(cd ../../../torchdata && uv pip install -e .)
	(cd ../../../torchaudio && uv pip install -e . --no-build-isolation)
	(cd ../../../FBGEMM/fbgemm_gpu && uv pip install -r requirements.txt && uv pip install -e . --no-build-isolation)
	(cd ../../../torchrec && uv pip install -e .)
	(cd ../../../detectron2 && uv pip install -e . --no-build-isolation)
	(cd ../../../torchbenchmark && python install.py --continue_on_fail $(if $(TORCHBENCH_MODELS),models $(TORCHBENCH_MODELS)))
	uv pip uninstall torchrec-nightly fbgemm-gpu-nightly
