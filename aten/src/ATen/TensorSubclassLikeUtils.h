#pragma once
#include <ATen/ATen.h>

namespace at {

// Note [Tensor-subclass-like Tensors]
// Tensor-subclass-like is defined as:
// - a Tensor subclass (via __torch_dispatch__ in Python or extending
//   TensorImpl in C++)
// - anything else that shares the same perils as Tensor subclasses.
//   For example, many Tensor subclasses do not have storage and meta Tensors
//   do not have storage either, so meta Tensors belong here.
//
// We should ensure that PyTorch internals supports Tensor-subclass-like
// objects. In particular, Tensor-subclass-like objects struggle with two
// classes of operations that are problematic for Tensor subclasses:
// 1. Because some Tensor subclasses do not have storage, .item() or
//    .data_ptr() calls are not good.
// 2. Certain in-place operations can eliminate the typing of the Tensor
//    subclass. For example:
//    >>> torch.zeros(input.sizes(), grad.options()).diag().copy_(input)
//    If input is a Tensor subclass, then the above ends up either erroring out
//    or returning a regular non-Tensor-subclass Tensor!

constexpr auto kFunctorchWrappedTensors = DispatchKeySet({
    DispatchKey::FuncTorchGradWrapper,
    DispatchKey::FuncTorchBatched});

constexpr auto kTensorSubclassLike = kFunctorchWrappedTensors | DispatchKeySet({
    DispatchKey::Batched,
    DispatchKey::SparseCPU,
    DispatchKey::SparseCUDA,
    DispatchKey::SparseCsrCPU,
    DispatchKey::SparseCsrCUDA,
    DispatchKey::Meta,
    DispatchKey::Python});

inline bool isTensorSubclassLike(const Tensor& tensor) {
  auto key_set = tensor.unsafeGetTensorImpl()->key_set();
  return !(key_set & kTensorSubclassLike).empty();
}

inline bool areAnyTensorSubclassLike(TensorList tensors) {
  return std::any_of(tensors.begin(), tensors.end(), isTensorSubclassLike);
}

inline bool areAnyOptionalTensorSubclassLike(const c10::List<c10::optional<Tensor>>& tensors) {
  return std::any_of(tensors.begin(), tensors.end(), [](const optional<Tensor>& opt_tensor) {
    return (opt_tensor.has_value() && isTensorSubclassLike(opt_tensor.value()));
    });
}

}
