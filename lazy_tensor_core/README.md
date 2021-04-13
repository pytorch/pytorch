# Lazy Tensors Core

Provides infrastructure to connect the [PyTorch deep learning framework](https://pytorch.org/)
to deep learning acelerators, with a focus on training. This project is derived from [PyTorch/XLA](https://github.com/pytorch/xla), which introduced the approach for training on Google TPUs, but instead aims for generality across vendors.

## <a name="API"></a> API & Best Practices

In general, lazy tensors follow PyTorch APIs. Some additional lazy_tensor_core specific APIs are available at:

[Documentation for master branch](https://pytorch.org/ltc/master)

See the [API Guide](API_GUIDE.md) for best practices when writing networks that
run on deep learning accelerators.

## <a name="Troubleshooting"></a> Troubleshooting

If an accelerator integration using this infrastructure isn't performing as expected,
see the [troubleshooting guide](TROUBLESHOOTING.md), which has suggestions for
debugging and optimizing your network(s).

## <a name="Feedback"></a> Providing Feedback

The PyTorch team is always happy to hear from users and OSS contributors!
The best way to reach out is by filing an issue on this Github. Questions,
bug reports, feature requests, build issues, etc. are all welcome!

## <a name="Contributing"></a> Contributing

See the [contribution guide](CONTRIBUTING.md).

## Disclaimer 
This repository is jointly operated and maintained by Google, Facebook and a number of individual contributors listed in the [CONTRIBUTORS](https://github.com/pytorch/ltc/graphs/contributors) file.
