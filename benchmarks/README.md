# PyTorch Benchmarks

NOTE: This folder is currently work in progress.

This folder contains scripts that produce reproducible timings of various PyTorch features.

It also provides mechanisms to compare PyTorch with other frameworks.

## Setup environment
Make sure you're on a machine with CUDA, torchvision, and pytorch installed. Install in the following order:
```
# Install torchvision. It comes with the pytorch stable release binary
conda install pytorch torchvision -c pytorch

# Install the latest pytorch master from source.
# It should supersede the installation from the release binary.
cd $PYTORCH_HOME
python setup.py build develop

# Check the pytorch installation version
python -c "import torch; print(torch.__version__)"
```

## Benchmark List

Please refer to each subfolder to discover each benchmark suite

* [Fast RNNs benchmarks](fastrnns/README.md)

