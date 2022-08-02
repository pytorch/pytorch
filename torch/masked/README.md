# maskedtensor

**Warning: This is a prototype library that is actively under development. If you have suggestions or potential use cases that you'd like addressed, please
open a Github issue; we welcome any thoughts, feedback, and contributions!**

MaskedTensor is a prototype library that is part of the [PyTorch](https://pytorch.org/) project and is an extension of `torch.Tensor` that provides the ability to mask out the value for any given element. Elements with masked out values are ignored during computation and give the user access to advanced semantics such as masked reductions, safe softmax, masked matrix multiplication, filtering NaNs, and masking out certain gradient values.
### Notebooks

For an introduction and instructions on how to use MaskedTensors and what they are useful for, there are a number of tutorials on the [MaskedTensor website](https://pytorch.org/maskedtensor/main/index.html).

## License

maskedtensor is licensed under BSD 3-Clause
