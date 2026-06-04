#pragma once

#include <ATen/core/Tensor.h>
#include <rust/cxx.h>
#include <cstdint>

// Annotate a raw pointer return that may be null so nullability checkers (e.g.
// the internal NullableReturn lint) recognize it. Mirrors folly's
// FOLLY_NULLABLE without depending on folly (not available in OSS builds): emit
// Clang's `_Nullable` extension and silence -Wnullability-extension as folly
// does; expand to nothing on other compilers. Kept on one line so clang-format
// does not reflow the escaped-newline macro.
#if defined(__clang__)
#define TORCH_RUST_NULLABLE _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wnullability-extension\"") _Nullable _Pragma("clang diagnostic pop")
#else
#define TORCH_RUST_NULLABLE
#endif

namespace torch::rust {

using Tensor = at::Tensor;

// Returns a borrowed pointer to the at::Tensor owned by the THPVariable, or
// nullptr if py_obj is not a torch.Tensor. Lifetime is tied to the Python
// object — the caller must keep the Python reference alive while using it.
const Tensor* TORCH_RUST_NULLABLE tensor_from_pyobject(
    std::uintptr_t py_obj) noexcept;

int64_t tensor_dim(const Tensor& t);
int64_t tensor_numel(const Tensor& t);
int64_t tensor_size_at(const Tensor& t, int64_t dim);
int64_t tensor_stride_at(const Tensor& t, int64_t dim);

::rust::Slice<const int64_t> tensor_sizes(const Tensor& t);
::rust::Slice<const int64_t> tensor_strides(const Tensor& t);

bool tensor_is_contiguous(const Tensor& t);
bool tensor_is_cpu(const Tensor& t);
bool tensor_is_cuda(const Tensor& t);
bool tensor_defined(const Tensor& t);
bool tensor_requires_grad(const Tensor& t);

} // namespace torch::rust
