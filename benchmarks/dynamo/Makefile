# Makefile for the dashboard setup
PIP ?= python -m pip

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
	(cd ../../../torchvision    && git fetch && git checkout d23a6e1664d20707c11781299611436e1f0c104f && git submodule update --init --recursive)
	(cd ../../../torchtext      && git fetch && git checkout b0ebddc648d279826089db91775375221777a2db && git submodule update --init --recursive)
	(cd ../../../torchaudio     && git fetch && git checkout bccaa454a54c3c648697cc2f46a4fb0500b1f01b && git submodule update --init --recursive)
	(cd ../../../FBGEMM         && git fetch && git checkout de731af65b4f04696e85c729e3282450b51b95fd && git submodule update --init --recursive)
	(cd ../../../torchrec       && git fetch && git checkout 6cd9fd362514d14ebb9ed51314c62ac1e1e2bbf2 && git submodule update --init --recursive)
	(cd ../../../detectron2     && git fetch && git checkout HEAD && git submodule update --init --recursive)
	(cd ../../../torchbenchmark && git fetch && git checkout 373ffb19dc470f4423a3176a4133f8f4b3cdb5bd && git submodule update --init --recursive)

build-deps: clone-deps
	# conda env remove --name torchdynamo
	# conda create --name torchdynamo -y python=3.8
	# conda activate torchdynamo
	conda install -y astunparse numpy scipy ninja pyyaml mkl mkl-include setuptools cmake \
		typing-extensions requests protobuf numba cython scikit-learn
	conda install -y -c pytorch magma-cuda116
	conda install -y -c conda-forge librosa
	(cd ../../../torchvision && python setup.py clean && python setup.py develop)
	# (cd ../../../torchdata && python setup.py install)
	(cd ../../../torchtext && python setup.py clean && python setup.py develop)
	(cd ../../../torchaudio && python setup.py clean && python setup.py develop)
	(cd ../../../FBGEMM/fbgemm_gpu && pip install -r requirements.txt && python setup.py clean && python setup.py develop)
	(cd ../../../torchrec && python setup.py clean && python setup.py develop)
	(cd ../../../detectron2  && python setup.py clean && python setup.py develop)
	(cd ../../../torchbenchmark && python install.py --continue_on_fail)
	pip uninstall -y torchrec-nightly fbgemm-gpu-nightly
