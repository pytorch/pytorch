#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <rust/cxx.h>
#include <torch/csrc/python_headers.h>
#include <cstdint>
#include <exception>

namespace rust::behavior {

// What a fallible function of the bridge (one that returns Result in
// src/bindings/ffi.rs) throws becomes its Err: every exception, without the
// text of a C++ backtrace, and with the Python error it may have been made
// of dropped.
template <typename Try, typename Fail>
static void trycatch(Try&& func, Fail&& fail) noexcept try {
  func();
} catch (...) {
  if (PyGILState_Check() && PyErr_Occurred()) {
    PyErr_Clear();
  }
  try {
    throw;
  } catch (const c10::Error& e) {
    fail(e.what_without_backtrace());
  } catch (const std::exception& e) {
    fail(e.what());
  } catch (...) {
    fail("an exception that is no std::exception");
  }
}

} // namespace rust::behavior

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
bool torch_function_mode_active() noexcept;

bool grad_mode_enabled() noexcept;

// Defined by the cxx bridge (src/bindings/ffi.rs).
struct Word;
// The ambient word: the state of the thread and the process a region is for,
// packed. A region is entered only under the word of its trace. In `a`:
// autocast (on, and its dtype) for cpu, cuda, xpu and mps, the default
// dtype, the three deterministic-algorithms flags, cuDNN enabled / benchmark
// / deterministic, mkldnn enabled, grad mode, inference mode, and the
// current CUDA device. In `b`: the float32 precision of every (backend,
// operator) pair, as resolved, and oneDNN's allow_tf32. False, and nothing
// of `out` to be read, if any of it cannot be read or has a value the word
// has no room for: no trace then, and no region entered. Needs the GIL.
bool region_word(Word& out) noexcept;

// Defined by the cxx bridge (src/bindings/ffi.rs).
struct AmbientModes;

// What an own trace of the engine has on while the traced body runs: how
// many dispatch modes and torch function modes there are, and whether
// torch.fx is tracing (0: no, 1: yes, 2: it cannot be said, which is never
// neutral). Needs the GIL.
void region_modes(AmbientModes& out) noexcept;
// 0 if the state outside the word is neutral, else which item is not (see
// src/exec/ambient.rs): a dispatch mode or a torch function mode beyond
// `own`'s, a functorch transform, a forward-AD level, saved-tensor hooks,
// JIT tracing, fx tracing other than `own`'s, anomaly mode, autocast for a
// device type the word does not name; 255 if it could not be asked. Needs
// the GIL.
std::uint8_t region_not_neutral(const AmbientModes& own) noexcept;
// Sets grad mode; returns what it was.
bool set_grad_mode(bool on) noexcept;
// The version counter in-place ops bump (Tensor._version); -1 for a tensor
// that has none (an inference tensor, an undefined one).
int64_t tensor_version(const Tensor& t) noexcept;
// Where t's first element is or, for a tensor without elements, would be
// (the storage's address and t's offset in it), and into `storage` which
// storage that is: what tensor_span gives as `low` and `storage`, without
// the walk over the sizes. Both 0 if t has no memory of its own to point at
// (undefined, not strided, a Python-dispatch tensor). Read without asking
// for a mutable pointer, so a copy-on-write storage stays one.
std::uintptr_t tensor_home(const Tensor& t, std::uintptr_t& storage) noexcept;
// Defined by the cxx bridge (src/bindings/ffi.rs).
struct TensorSpan;
// Where t's elements are: the storage and how many bytes into it the first
// starts, and the addresses from the first element to the end of the last,
// [low, high) (low == high for a tensor without elements: where its first
// would be), in the memory of which device type. False, and nothing of
// `out` to be read, for a tensor
// whose place cannot be said so: undefined, not strided, a Python-dispatch
// tensor, without a storage or with one that gives no address for its
// elements, or on a device type that is neither the CPU nor CUDA. Read
// without asking for a mutable pointer (see tensor_home).
bool tensor_span(const Tensor& t, TensorSpan& out) noexcept;
// Whether t's storage is marked copy-on-write. The mark is cleared by the
// first request for a mutable pointer to the storage, through whichever
// tensor: torch.vm.Unwritten's "nothing wrote here" (see torch/vm/_guards.py).
bool tensor_is_cow(const Tensor& t) noexcept;
// Whether a forward-mode AD level is open (as custom autograd Functions check).
bool forward_ad_active() noexcept;
// Whether saved_tensors_hooks would pack tensors saved for backward.
bool saved_tensor_hooks_active() noexcept;

// Defined by the cxx bridge (src/bindings/ffi.rs).
struct TensorFacts;

// The one definition of what a region compares of a tensor it is entered
// with (see src/exec/tensor_key.rs, which adds the Python class): fills in
// the dtype and the device (c10's enums) and, in `bits`, requires_grad (1),
// the conjugate bit (2), the negative bit (4) and whether it is an inference
// tensor (8); the sizes and strides are the tensor's own. False for a tensor
// that has no key and so enters no region: one that is not a plain dense
// strided tensor with concrete sizes.
bool tensor_key(const Tensor& t, TensorFacts& out) noexcept;
// Whether t has a key, and it is exactly this one.
bool tensor_key_is(
    const Tensor& t,
    const TensorFacts& want,
    ::rust::Slice<const int64_t> sizes,
    ::rust::Slice<const int64_t> strides) noexcept;

// -1, or false, for a tensor that does not say (see the bridge's rule in
// src/bindings/ffi.rs).
int64_t tensor_dim(const Tensor& t) noexcept;
int64_t tensor_numel(const Tensor& t) noexcept;
int64_t tensor_size_at(const Tensor& t, int64_t dim) noexcept;
int64_t tensor_stride_at(const Tensor& t, int64_t dim) noexcept;

// These two may throw (a nested tensor, symbolic sizes; a Python-dispatch
// tensor answers in Python): fallible in the bridge.
::rust::Slice<const int64_t> tensor_sizes(const Tensor& t);
::rust::Slice<const int64_t> tensor_strides(const Tensor& t);

bool tensor_is_contiguous(const Tensor& t) noexcept;
bool tensor_is_cpu(const Tensor& t) noexcept;
bool tensor_is_cuda(const Tensor& t) noexcept;
bool tensor_defined(const Tensor& t) noexcept;
bool tensor_requires_grad(const Tensor& t) noexcept;
bool tensor_is_leaf(const Tensor& t) noexcept;

// Defined by the cxx bridge (src/bindings/ffi.rs).
struct TensorMeta;

// What the CFG executor's type lattice records about a tensor, read without
// running Python or throwing: a Python-dispatch tensor may compute its device
// and sizes in Python, so only its dtype and requires_grad are filled in, and
// symbolic sizes are left out.
void tensor_meta(const Tensor& t, TensorMeta& out) noexcept;

// Short dtype names (f32, i64, ...), as printed for ops and types.
const char* dtype_short_name(at::ScalarType t);
::rust::String dtype_name(int8_t dtype) noexcept;
::rust::String device_name(int8_t type, int8_t index) noexcept;

} // namespace torch::rust
