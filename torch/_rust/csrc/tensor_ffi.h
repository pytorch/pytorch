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

// Like tensor_from_pyobject, but by the object's type alone, so it never runs
// Python code (isinstance can, through __instancecheck__ or a __class__
// property).
const Tensor* TORCH_RUST_NULLABLE tensor_by_type(std::uintptr_t py_obj) noexcept;

// Whether a __torch_function__ mode would intercept torch calls.
bool torch_function_mode_active();

bool grad_mode_enabled();
// The thread state a traced graph is specialized on beyond grad mode, packed:
// autocast (on, and its dtype) for cpu, cuda, xpu and mps, the default dtype,
// and the three deterministic-algorithms flags.
std::uint64_t region_thread_state();
// Sets grad mode; returns what it was.
bool set_grad_mode(bool on);
// The version counter in-place ops bump (Tensor._version).
int64_t tensor_version(const Tensor& t);
// Where t's first element is, 0 if t has no memory of its own to point at
// (undefined, not strided, a Python-dispatch tensor). Read without asking
// for a mutable pointer, so a copy-on-write storage stays one.
std::uintptr_t tensor_data_address(const Tensor& t) noexcept;
// Defined by the cxx bridge (src/bindings/ffi.rs).
struct TensorSpan;
// The storage t's elements are in, and the bytes of it they span, [low, high)
// from the storage's start (low == high for a tensor without elements). The
// storage is 0 for a tensor with no memory of its own to point at (see
// tensor_data_address). Nothing is asked of the storage's data.
void tensor_span(const Tensor& t, TensorSpan& out) noexcept;
// Whether a forward-mode AD level is open (as custom autograd Functions check).
bool forward_ad_active();
// Whether saved_tensors_hooks would pack tensors saved for backward.
bool saved_tensor_hooks_active();

// Whether t is a plain strided tensor with exactly this dtype, device, sizes
// and strides (c10's enums for dtype and device).
bool tensor_matches(
    const Tensor& t,
    int8_t dtype,
    int8_t device_type,
    int8_t device_index,
    ::rust::Slice<const int64_t> sizes,
    ::rust::Slice<const int64_t> strides) noexcept;

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

// Defined by the cxx bridge (src/bindings/ffi.rs).
struct TensorMeta;

// What the CFG executor's type lattice records about a tensor, read without
// running Python or throwing: a Python-dispatch tensor may compute its device
// and sizes in Python, so only its dtype and requires_grad are filled in, and
// symbolic sizes are left out.
void tensor_meta(const Tensor& t, TensorMeta& out) noexcept;

// Short dtype names (f32, i64, ...), as printed for ops and types.
const char* dtype_short_name(at::ScalarType t);
::rust::String dtype_name(int8_t dtype);
::rust::String device_name(int8_t type, int8_t index);

} // namespace torch::rust
