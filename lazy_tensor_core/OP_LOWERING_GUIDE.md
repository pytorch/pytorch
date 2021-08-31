# OP Lowering Guide

## Background
PyTorch wraps the C++ ATen tensor library that offers a wide range of operations implemented on GPU and CPU. Lazy tensors implement those operations on a virtual, lazy device and recover a graph computation by recording those operations instead of executing them right away. That graph can be compiled to native code for various physical devices (CPU, GPU, other hardware accelerators) by a back-end plugin. We bundle a TorchScript plugin to support vendors which already have compilers from TorchScript to native code, but also to get test coverage for the lazy tensors core functionality and serve as an example for other plugins, such as [XLA](https://github.com/pytorch/xla/tree/asuhan/xla_ltc_plugin).

"Lowering" defines the process of converting lazy tensors computation graphs, which are DAGs (directed acyclic graphs) of ATen operations as captured by executing a model. Plugins, which include the bundled TorchScript plugin, have their own lowering component to native code, but that’s beyond the scope of this documentation. Operations which don't have a lowering will automatically execute using their implementations in the PyTorch core, on CPU and GPU. However, doing so too often defeats the purpose of graph capture through lazy execution, since each such operation forces the evaluation of its inputs, thus fragmenting the captured graph. To maximize the potential of this approach, all operations used by a model should be lowered.

## Before you start
You should follow the [instructions](https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/lazy_tensor_core/QUICKSTART.md) to build the lazy tensors component from source.

## Understanding the operation
You can find the definition of the C++ ATen operations in [native_functions.yaml](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml). PyTorch operations can usually be mapped to [PyTorch tensor api](https://pytorch.org/docs/stable/index.html) easily. If that is not the case searching the PyTorch native implementation under [PyTorch repo](https://github.com/pytorch/pytorch) is recommended. The goal is to lower the PyTorch operations into a sequence of TorchScript instructions, which most of the time is simply [calling a builtin method](https://github.com/pytorch/pytorch/blob/b1daf83196dee499defe420fb1a90ad7da7ed05c/lazy_tensor_core/lazy_tensor_core/csrc/ts_backend/ts_node_lowering.cpp#L290).

## File structure
Files mentioned below live under the `lazy_tensor_core/lazy_tensor_core/csrc` and `lazy_tensor_core/lazy_tensor_core/csrc/ts_backend` folders, with the exception of `ts_native_functions.yaml`

1. `ts_native_functions.yaml` contains the list of all operators that are lowered. Each operator name must directly match a pytorch operator listed in [native_functions.yaml](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml). This file serves as the interface to adding support for new operators, and is an input to PyTorch's [codegen machinery](https://github.com/pytorch/pytorch/blob/master/tools/codegen/gen_backend_stubs.py). It generates the below 3 files: `LazyNativeFunctions.h`, `RegisterLazy.cpp`, and `RegisterAutogradLazy.cpp`
1. `LazyNativeFunctions.h` and `aten_ltc_ts_type.cpp` are entry points of PyTorch to the lazy tensors world, and contain the manually written lowerings for each operator. `LazyNativeFunctions.h` is auto-generated through a combination of `ts_native_functions.yaml` and the PyTorch core `native_functions.yaml` file, and contains declarations for kernels that need to be defined in `aten_ltc_ts_type.cpp`. The kernels written here need to construct `LazyTensor`s using the input `at::Tensor` and other parameters. The resulting `LazyTensor` needs to be converted back to the `at::Tensor` before returning to the PyTorch world.
1. `RegisterLazy.cpp` and `RegisterAutogradLazy.cpp` are auto-generated files that register all lowerings to the PyTorch Dispatcher. They also include auto-generated wrapper implementations of `out=` and `inplace` operators.
1. `aten_eager_fallback.h/.cpp` contain our boxed fallback implementation to CPU or GPU. The boxed fallback kernel will be used if a lowering is not explicitly defined in `ts_native_functions.yaml` + `aten_ltc_ts_type.cpp`, and the operator is not composite.
1. `tensor.h` contains the `LazyTensor` declarations. These declarations are usually a one to one mapping of the `at::Tensor` nodes we declared in `LazyNativeFunctions.h`
1. `tensor_methods.cpp` contains the implementations of `LazyTensor` methods defined in `tensor.h`. They are expressed in terms of `ir::Value`s, which are wrappers for nodes in the captured computation graph.
1. `ops/` directory contains definitions for the various node kinds in the computation graphs. All ops inherit from `ir::ops::Node`.
1. `ts_node_lowering.cpp` contains the main conversion logic that lowers Lazy Tensor operations (IR) into TorchScript operations (IR) for a particular operator, and also shape inferences. *Only* modify this file if the default-lowering and default-shape-inferring logic don't work for you. If that's the case, you will get exceptions when executing the operators.
    1. Normally, operators that not only accept operands but also extra parameters will need dedicated lowering, for example, _log_softmax. FYI, operators that work with default-lowering are, for example, addcdiv, sqrt, and etc.
    1. For shape inferences, usually the PyTorch.org documentation of the operator has well illustrated the forward pass. For the backward, it's normally the shape of the input (think about what gradients are). The native implementation under `aten/src/ATen/native` is always a good reference. To be noted, the new case might very well be folded into one of the existing shape inference rules under the switch table.

## Unit Test
C++ unit tests for tensor operations are in the `lazy_tensor_core/test/cpp/test_aten_ltc_ts_tensor.cpp` file. This verifies the TorchScript lazy tensors back-end against PyTorch CPU implementation. Some of these tests also check if the lowering we provide is actually called, by checking counters which track how many times an operation is hit, separating the hits between the provided lowering or the fallback.

## Tips
We have auto-generated wrapper implementations of `out=` and `inplace` operators for some operators in `RegisterLazy.cpp`. We only need to lower the base operator in this case. An example would be `lerp` operator which has 6 variants in `native_functions.yaml`, they are

```
  - lerp_.Scalar
  - lerp_.Tensor
  - lerp.Scalar_out
  - lerp.Tensor_out
  - lerp.Scalar
  - lerp.Tensor
```

and will generate function prototypes

```
at::Tensor lerp(const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight);
at::Tensor & lerp_(at::Tensor & self, const at::Tensor & end, const at::Scalar & weight);
at::Tensor lerp(const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight);
at::Tensor & lerp_out(const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight, at::Tensor & out);
at::Tensor & lerp_(at::Tensor & self, const at::Tensor & end, const at::Tensor & weight);
at::Tensor & lerp_out(const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight, at::Tensor & out);
```

in `LazyNativeFunctions.h` if we add all of them to the `ts_native_functions.yaml`. However if we only lower `lerp.Scalar` and `lerp.Tensor` and check `RegisterLazy.cpp`, we will see

```
namespace {

at::Tensor wrapper_Scalar_lerp(const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) {
    // No device check


  // DeviceGuard omitted
  return torch_lazy_tensors::lerp(self, end, weight);
}

} // anonymous namespace

at::Tensor & wrapper_Scalar_lerp_(at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) {
  auto wrapper_Scalar_lerp__tmp = wrapper_Scalar_lerp(self, end, weight);
  at::_copy_from(wrapper_Scalar_lerp__tmp, self);
  return self;
}

...
  m.impl("lerp_.Scalar",
  TORCH_FN(wrapper_Scalar_lerp_));

```

`lerp_.Scalar` will use our `lerp.Scalar` implementation without us providing explictly lowering.

For each node we need to pass an `ir::OpKind`. Here is an ([example](https://github.com/pytorch/pytorch/blob/700731c40bbc47faff14d49e77f8322ebd1c2d5b/lazy_tensor_core/lazy_tensor_core/csrc/ops/var_mean.cpp#L10)). You can find the `OpKind` definition in [aten_interned_strings.h](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/aten_interned_strings.h) or [interned_strings.h](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/interned_strings.h). If the aten symbol is missing, you can submit a PR like [this](https://github.com/pytorch/pytorch/pull/36851).

It's a double edge sword to refer to the XLA's implementation as sometimes it gives good hints on how things should be lowered if the operator shares the same implementation in both the TS and XLA backends, or sometimes it shows a different direction if the operator has a very XLA specific implementation.
