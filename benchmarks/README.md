# PyTorch Benchmarks

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

Please refer to each subfolder to discover each benchmark suite. Links are provided where descriptions exist:

* [Fast RNNs](fastrnns/README.md)
* [Dynamo](dynamo/README.md)
* [Functional autograd](functional_autograd_benchmark/README.md)
* [Instruction counts](instruction_counts/README.md)
* [Operator](operator_benchmark/README.md)
* [Overrides](overrides_benchmark/README.md)
* [Sparse](sparse/README.md)
* [Tensor expression](tensorexpr/HowToRun.md)
