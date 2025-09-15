# AMP(Automatic Mixed Precision) Mechanism

## 1. Overview

Automatic Mixed Precision (AMP) enables the use of both single precision (32-bit) and half precision (16-bit) floating point types during training or inference. Find more details in [here](https://docs.pytorch.org/docs/stable/amp.html).

Key components include:

- [**Autocast**](https://docs.pytorch.org/docs/stable/amp.html#autocasting): Automatically casts operations to lower-precision (e.g., float16 or bfloat16) to improve performance while maintaining accuracy.
- [**Gradient Scaling**](https://docs.pytorch.org/docs/stable/amp.html#gradient-scaling): Dynamically scales gradients during backpropagation to prevent underflow when training with mixed precision.


## 2. `torch.amp.autocast` Integration

### 2.1 Python Integration

**Register Supported Data Type**

Implement the `get_amp_supported_dtype` method to return the data types supported by the accelerator in the AMP context.

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/amp/__init__.py
    :language: python
    :start-after: LITERALINCLUDE START: GET_AMP_SUPPORTED_DTYPE
    :end-before: LITERALINCLUDE END: GET_AMP_SUPPORTED_DTYPE
    :linenos:
```

### 2.2 C++ Integration

**Kernel Registration**

This section shows how AMP registers autocast kernels for the `AutocastPrivateUse1` dispatch key.

- Register a fallback that makes unhandled ops fall through to their normal implementations.
- Register specific aten kernels under `AutocastPrivateUse1` using the `KERNEL_PRIVATEUSEONE` helper macro, which maps an op to the desired precision implementation (with enum `at::autocast::CastPolicy`)

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/amp/autocast_mode.cpp
    :language: python
    :start-after: LITERALINCLUDE START: AMP_REGISTER_FALLBACK
    :end-before: LITERALINCLUDE END: AMP_REGISTER_FALLBACK
    :linenos:

.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/amp/autocast_mode.cpp
    :language: python
    :start-after: LITERALINCLUDE START: AMP_REGISTER_KERNEL
    :end-before: LITERALINCLUDE END: AMP_REGISTER_KERNEL
    :linenos:
```

### 2.3 Casting Strategy: `at::autocast::CastPolicy`

[`CastPolicy`](https://github.com/pytorch/pytorch/blob/09587daf8c9f21f5340f73921ce5f23d1a4a4572/aten/src/ATen/autocast_mode.h#L416-L438) serves as a label for type handling rules. Each enum value represents the casting needs of a class of operators, ensuring consistent handling of precision-sensitive vs. performance-oriented operations.

| Policy | Explain | Typical Use Cases |
|---|---|---|
| **`lower_precision_fp`** | Cast all inputs to the device’s preferred low-precision dtype. | Ops that tolerate reduced precision (e.g., convolution, linear layers)|
| **`fp32`** | Force all inputs to `at::kFloat` (FP32) to avoid low-precision errors. | Precision-critical ops (e.g., loss, optimizer steps)|
| **`fp32_set_opt_dtype`** | Handle ops with `std::optional<ScalarType>` args (e.g., `softmax`) | Ensure execution in FP32, while respecting user-specified output dtype if provided. If not provided, set output dtype to FP32 automatically. |                                                                                                                                                                                        |
| **`fp32_append_dtype`**  | Handle overloaded ops (e.g., `norm`)| Wraps overloads without `dtype` by appending `fp32` as an extra argument and redirecting to the overload with a dtype parameter.||
| **`promote`** | Automatically promote all inputs to the “widest” dtype before execution. | Ops receiving inputs with different dtypes (e.g., adding FP16 + FP32 tensors) |
