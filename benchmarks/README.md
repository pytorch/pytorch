# PyTorch Benchmarks

This folder contains scripts that produce reproducible timings of various PyTorch features.

It also provides mechanisms to compare PyTorch with other frameworks.

## Setup environment
Make sure you're on a machine with CUDA, torchvision, and PyTorch installed.
For the latest installation instructions, please refer to the [PyTorch installation page](https://pytorch.org/get-started/locally).

Install in the following order:
```
# Install torchvision. It comes with the PyTorch stable release binary.
pip install torch torchvision

# Install the latest PyTorch master from source.
# It should supersede the installation from the release binary.
cd $PYTORCH_HOME
python setup.py build develop

# Check the PyTorch installation version.
python -c "import torch; print(torch.__version__)"
```

## Benchmark List

Please refer to each subfolder to discover each benchmark suite. Links are provided where descriptions exist:

* [Fast RNNs](fastrnns/README.md)
* [Dynamo](dynamo/README.md)
* [Functional autograd](functional_autograd_benchmark/README.md)
* [Instruction counts](instruction_counts/README.md)
* [Operator](operator_benchmark/README.md)
* [Overrides](overrides_benchmark/README.md)
* [Sparse](sparse/README.md)
* [Tensor expression](tensorexpr/HowToRun.md)
