# maskedtensor

**Warning: This is a prototype library that is actively under development. If you have suggestions or potential use cases that you'd like addressed, please
open a Github issue; we welcome any thoughts, feedback, and contributions!**

MaskedTensor is a prototype library that is part of the [PyTorch](https://pytorch.org/) project and is an extension of `torch.Tensor` that provides the ability to mask out the value for any given element. Elements with masked out values are ignored during computation and give the user access to advanced semantics such as masked reductions, safe softmax, masked matrix multiplication, filtering NaNs, and masking out certain gradient values.

## Installation

### Binaries

To install the official MaskedTensor via pip, use the following command:

```
pip install maskedtensor
```

For the dev (unstable) nightly version that contains the most recent features, please replace `maskedtensor` with `maskedtensor-nightly`.

Note that MaskedTensor requires PyTorch >= 1.11, which you can get on the [the main website](https://pytorch.org/get-started/locally/)

### From Source

To install from source, you will need Python 3.7 or later, and we highly recommend that you use an Anaconda environment. Then run:

```
python setup.py develop
```

## Documentation

Please find documentation on the [MaskedTensor Website](https://pytorch.org/maskedtensor/main/index.html).

### Building documentation

Please follow the instructions in the [docs README](https://github.com/pytorch/maskedtensor/tree/main/docs).

### Notebooks

For an introduction and instructions on how to use MaskedTensors and what they are useful for, there are a nubmer of tutorials on the [MaskedTensor website](https://pytorch.org/maskedtensor/main/index.html).

## License

maskedtensor is licensed under BSD 3-Clause
