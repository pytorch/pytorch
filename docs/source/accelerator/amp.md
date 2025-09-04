# AMP Integration

## 1. Overview

Automatic Mixed Precision (AMP) enables the use of both single precision (32-bit) and half precision (16-bit) floating point types during training or inference. Find more details in [here](https://docs.pytorch.org/docs/stable/amp.html).

Key components include:

- **Autocast**: Automatically casts operations to lower-precision (e.g., float16 or bfloat16) to improve performance while maintaining accuracy. 
- **GradScaler**: Dynamically scales gradients during backpropagation to prevent underflow when training with mixed precision.


## 2. `torch.amp.autocast` Integration

### 2.1 Kernel Registration

This section shows how AMP registers autocast kernels for the `AutocastPrivateUse1` dispatch key. 

- Register a fallback that makes unhandled ops fall through to their normal implementations. 
- Register specific aten kernels under `AutocastPrivateUse1` using the `KERNEL_PRIVATEUSEONE` helper macro, which maps an op to the desired precision implementation (`CastPolicy`)


```cpp
#include <ATen/autocast_mode.h>

TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
// lower_precision_fp
KERNEL_PRIVATEUSEONE(mm, lower_precision_fp)
// fp32
KERNEL_PRIVATEUSEONE(layer_norm, fp32)
// fp32_set_opt_dtype
KERNEL_PRIVATEUSEONE(prod, fp32_set_opt_dtype)
}
```

### 2.2 Casting Strategy: `CastPolicy`

[`CastPolicy`](https://github.com/pytorch/pytorch/blob/09587daf8c9f21f5340f73921ce5f23d1a4a4572/aten/src/ATen/autocast_mode.h#L416-L438) serves as a label for type handling rules. Each enum value represents the casting needs of a class of operators, ensuring consistent handling of precision-sensitive vs. performance-oriented operations.

| Policy | Explain | Typical Use Cases |
|---|---|---|
| **`lower_precision_fp`** | Cast all inputs to the device’s preferred low-precision dtype. | Ops that tolerate reduced precision (e.g., convolution, linear layers)|
| **`fp32`** | Force all inputs to `at::kFloat` (FP32) to avoid low-precision errors. | Precision-critical ops (e.g., loss, optimizer steps)|
| **`fp32_set_opt_dtype`** | Handle ops with `std::optional<ScalarType>` args (e.g., `softmax`) | Ensure execution in FP32, while respecting user-specified output dtype if provided. If not provided, set output dtype to FP32 automatically. |                                                                                                                                                                                        |
| **`fp32_append_dtype`**  | Handle overloaded ops (e.g., `norm`)| Wraps overloads without `dtype` by appending `fp32` as an extra argument and redirecting to the overload with a dtype parameter.||
| **`promote`** | Automatically promote all inputs to the “widest” dtype before execution. | Ops receiving inputs with different dtypes (e.g., adding FP16 + FP32 tensors) |

