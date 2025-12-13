# Automatic Mixed Precision

## Background

Automatic Mixed Precision (AMP) enables the use of both single precision (32-bit) and half precision (16-bit) floating point types during training or inference.

Key components include:

- [**Autocast**](https://docs.pytorch.org/docs/stable/amp.html#autocasting): Automatically casts operations to lower-precision (e.g., float16 or bfloat16) to improve performance while maintaining accuracy.
- [**Gradient Scaling**](https://docs.pytorch.org/docs/stable/amp.html#gradient-scaling): Dynamically scales gradients during backpropagation to prevent underflow when training with mixed precision.

## Design

### Casting Strategy

The [`CastPolicy`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/autocast_mode.h#L416-L438) is used to define type conversion rules. Each enum value represents a set of type conversion requirements for a group of operators, ensuring consistent handling of operations that prioritize either precision or performance.

| Policy                   | Explanation                                                                          |
| :---                     | :---                                                                                 |
| **`lower_precision_fp`** | Cast all inputs to `lower_precision_fp` before execute the op.                       |
| **`fp32`**               | Cast all inputs to `at::kFloat` before running the op.                               |
| **`fp32_set_opt_dtype`** | Execution in `at::kFloat`, while respecting user-specified output dtype if provided. |
| **`fp32_append_dtype`**  | Append at::kFloat to the args and redispatch to the type-aware overload              |
| **`promote`**            | Promote all inputs to the “widest” dtype before execution.                           |

### Operators Lists

PyTorch defines a general list of operators for each of casting strategies mentioned above, as a reference for developers of new accelerators.

| Policy                   | Operators List                                                                                    |
| :---                     | :---                                                                                              |
| **`lower_precision_fp`** | [List Link](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/autocast_mode.h#L819-L852) |
| **`fp32`**               | [List Link](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/autocast_mode.h#L854-L912) |
| **`fp32_set_opt_dtype`** | [List Link](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/autocast_mode.h#L914-L931) |
| **`fp32_append_dtype`**  | [List Link](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/autocast_mode.h#L933-L958) |
| **`promote`**            | [List Link](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/autocast_mode.h#L960-L971) |

## Implementation

### Python Integration

Implement the `get_amp_supported_dtype` method to return the data types supported by the new accelerator in the AMP context.

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/amp/__init__.py
    :language: python
    :start-after: LITERALINCLUDE START: AMP GET_SUPPORTED_DTYPE
    :end-before: LITERALINCLUDE END: AMP GET_SUPPORTED_DTYPE
    :linenos:
```

### C++ Integration

This section shows how AMP registers autocast kernels for the `AutocastPrivateUse1` dispatch key.

- Register a fallback that makes unhandled ops fall through to their normal implementations.
- Register specific aten kernels under `AutocastPrivateUse1` using the `KERNEL_PRIVATEUSEONE` helper macro, which maps an op to the desired precision implementation (with enum `at::autocast::CastPolicy`)

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/amp/autocast_mode.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: AMP FALLTHROUTH
    :end-before: LITERALINCLUDE END: AMP FALLTHROUTH
    :linenos:

.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/amp/autocast_mode.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: AMP IMPL
    :end-before: LITERALINCLUDE END: AMP IMPL
    :emphasize-lines: 3,6,8-10
    :linenos:
```
