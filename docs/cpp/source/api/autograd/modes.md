---
myst:
  html_meta:
    description: Gradient mode guards in PyTorch C++ — NoGradGuard and InferenceMode for disabling gradient computation.
    keywords: PyTorch, C++, NoGradGuard, InferenceMode, no_grad, inference, RAII guard
---

# Gradient Modes

PyTorch provides RAII guards to control gradient computation behavior.

## NoGradGuard

```{cpp:class} torch::NoGradGuard

RAII guard that disables gradient computation within its scope.
```

```{cpp:function} NoGradGuard()

Disables gradient computation.
```

```{cpp:function} ~NoGradGuard()

Restores previous gradient mode.
```

**Example:**

```cpp
{
    torch::NoGradGuard no_grad;
    // No gradients computed in this scope
    auto result = model->forward(input);
}
```

## InferenceMode

`c10::InferenceMode` is a RAII guard analogous to `NoGradMode` designed for use
when you are certain your operations will have no interactions with autograd
(e.g., model inference). Compared to `NoGradMode`, code run under this mode gets
better performance by disabling autograd-related work like view tracking and version
counter bumps. However, tensors created inside `InferenceMode` have more limitations
when interacting with the autograd system.

```{cpp:class} c10::InferenceMode

RAII guard that enables inference mode for optimized inference.
This is more efficient than NoGradGuard for inference-only workloads.
```

```{cpp:function} explicit InferenceMode(bool enabled = true)

Enables or disables inference mode.
```

**Inference Tensors:**

`InferenceMode` can be enabled for a given block of code. Inside `InferenceMode`,
all newly allocated (non-view) tensors are marked as inference tensors. Inference tensors:

- Do not have a version counter, so an error will be raised if you try to read their version
  (e.g., because you saved this tensor for backward).
- Are immutable outside `InferenceMode`. An error will be raised if you try to:

  - Mutate their data outside InferenceMode.
  - Mutate them to `requires_grad=True` outside InferenceMode.
  - To work around this, make a clone outside `InferenceMode` to get a normal tensor before mutating.

A non-view tensor is an inference tensor if and only if it was allocated inside `InferenceMode`.
A view tensor is an inference tensor if and only if it is a view of an inference tensor.

**Performance Guarantees:**

Inside an `InferenceMode` block:

- Like `NoGradMode`, all operations do not record `grad_fn` even if their inputs have
  `requires_grad=True`. This applies to both inference tensors and normal tensors.
- View operations on inference tensors do not perform view tracking. View and non-view
  inference tensors are indistinguishable.
- Inplace operations on inference tensors are guaranteed not to do a version bump.

For more implementation details, see the [RFC-0011-InferenceMode](https://github.com/pytorch/rfcs/pull/17).

**Basic Example:**

```cpp
{
    c10::InferenceMode guard;
    // Optimized inference without gradient tracking
    auto result = model->forward(input);
}
```

**Inference Workload Example:**

```cpp
c10::InferenceMode guard;
model.load_jit(saved_model);
auto inputs = preprocess_tensors(data);
auto out = model.forward(inputs);
auto outputs = postprocess_tensors(out);
```

**Nested InferenceMode:**

Unlike some other guards, `InferenceMode` can be nested with different enabled/disabled states:

```cpp
{
    c10::InferenceMode guard(true);
    // InferenceMode is on
    {
        c10::InferenceMode guard(false);
        // InferenceMode is off
    }
    // InferenceMode is on
}
// InferenceMode is off
```

## InferenceMode vs NoGradMode

`InferenceMode` is preferred over `NoGradMode` for pure inference workloads because
it provides better performance. Key differences:

- Both guards affect tensor execution to skip work not related to inference, but
  `InferenceMode` also affects tensor creation while `NoGradMode` doesn't.
- Tensors created inside `InferenceMode` are marked as inference tensors with
  certain limitations that apply after exiting `InferenceMode`.
- `InferenceMode` can be nested with enabled/disabled states.

## Migrating from AutoNonVariableTypeMode

The legacy `AutoNonVariableTypeMode` guard (now renamed to
`AutoDispatchBelowADInplaceOrView`) was commonly used for inference workloads
but is unsafe — it can silently bypass safety checks and produce wrong results.

- **For inference-only workloads** (e.g. loading a pretrained JIT model and
  running inference in C++ runtime), use `c10::InferenceMode` as a drop-in
  replacement. It preserves the performance characteristics while providing
  correctness guarantees.

- **For custom autograd kernels** that need to redispatch below the Autograd
  dispatch key, use `AutoDispatchBelowADInplaceOrView` instead:

  ```cpp
  class ROIAlignFunction : public torch::autograd::Function<ROIAlignFunction> {
   public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::Variable& input,
        const torch::autograd::Variable& rois,
        double spatial_scale, int64_t pooled_height,
        int64_t pooled_width, int64_t sampling_ratio, bool aligned) {
      ctx->saved_data["spatial_scale"] = spatial_scale;
      ctx->save_for_backward({rois});
      at::AutoDispatchBelowADInplaceOrView guard;
      auto result = roi_align(input, rois, spatial_scale,
          pooled_height, pooled_width, sampling_ratio, aligned);
      return {result};
    }
  };
  ```
