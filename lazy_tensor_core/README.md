# Lazy Tensors Core

Provides infrastructure to connect the [PyTorch deep learning framework](https://pytorch.org/)
to deep learning accelerators, with a focus on training. This project is derived from [PyTorch/XLA](https://github.com/pytorch/xla),
which introduced the approach for training on Google TPUs, but instead aims for generality across vendors.

## <a name="API"></a> API & Best Practices

In general, lazy tensors follow PyTorch APIs. Additional, specific APIs will be available,
including more general versions of the ones available for [PyTorch/XLA](https://pytorch.org/xla/master).

See the [API Guide](API_GUIDE.md) for best practices when writing networks that
run on deep learning accelerators.

## Adding Vendor Backends

Executing a computation for tensors on the lazy device requires a vendor backend.
If such a backend isn't implemented and registered, attempting to run such a
computation will fail with an error message indicating the missing implementation.

The `BackendImplInterface` coordinates the several major pieces required to implement
a vendor backend. A `BackendRegistrar` helper is provided to register the backend
once it's implemented. The aforementioned major pieces of a backend are defined by
the following interfaces:

* `NodeLowering`, which provides a way to lower PyTorch tensor operations to code
  for the accelerator.
* `Computation`, which defines tensor transfers to and from the accelerator
  memory and launching computations.
* `LoweringContext`, which provides state tracking to be used by code generation.

The current interfaces are subject to frequent changes and improvements as we learn
from vendor needs. We recommend tracking them and adjusting as needed until we
achieve a high degree of stability.

A re-implementation of PyTorch/XLA using this architecture [is available](https://github.com/pytorch/xla/tree/asuhan/xla_ltc_plugin),
reusing large parts of upstream PyTorch/XLA. Besides the in-line documentation
in the interfaces mentioned above, it provides a realistic example of a vendor
plug-in.

## <a name="Troubleshooting"></a> Troubleshooting

If an accelerator integration using this infrastructure isn't performing as expected,
see the [troubleshooting guide](TROUBLESHOOTING.md), which has suggestions for
debugging and optimizing your network(s).
