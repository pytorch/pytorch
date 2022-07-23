#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/PyInterpreter.h>

namespace c10 {
namespace impl {

static std::string noop_name_fn(const PyInterpreter*) {
  return "<unloaded interpreter>";
}

static void noop_decref_fn(const PyInterpreter*, PyObject*, bool) {
  // no-op
}

static c10::intrusive_ptr<TensorImpl> noop_detach_fn(
    const PyInterpreter*,
    const TensorImpl*) {
  TORCH_INTERNAL_ASSERT(
      0,
      "attempted to detach (shallow_copy_and_detach) Tensor with nontrivial PyObject after corresponding interpreter died");
}

static void noop_dispatch_fn(
    const PyInterpreter*,
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  TORCH_INTERNAL_ASSERT(
      0,
      "attempted to dispatch (__torch_dispatch__) an operator on Tensor with nontrivial PyObject after corresponding interpreter died");
}

static bool noop_is_contiguous_fn(const PyInterpreter*, const TensorImpl*) {
  TORCH_INTERNAL_ASSERT(
      0,
      "attempted to call `is_contiguous` on Tensor with nontrivial PyObject after corresponding interpreter died");
}

static c10::Device noop_device_fn(const PyInterpreter*, const TensorImpl*) {
  TORCH_INTERNAL_ASSERT(
      0,
      "attempted to call `device` on Tensor with nontrivial PyObject after corresponding interpreter died");
}

static int64_t noop_dim_fn(const PyInterpreter*, const TensorImpl*) {
  TORCH_INTERNAL_ASSERT(
      0,
      "attempted to call `dim` on Tensor with nontrivial PyObject after corresponding interpreter died");
}

static c10::IntArrayRef noop_strides_fn(
    const PyInterpreter*,
    const TensorImpl*) {
  TORCH_INTERNAL_ASSERT(
      0,
      "attempted to call `strides` on Tensor with nontrivial PyObject after corresponding interpreter died");
}

static c10::IntArrayRef noop_sizes_fn(const PyInterpreter*, const TensorImpl*) {
  TORCH_INTERNAL_ASSERT(
      0,
      "attempted to call `sizes` on Tensor with nontrivial PyObject after corresponding interpreter died");
}

static c10::SymIntArrayRef noop_sym_sizes_fn(
    const PyInterpreter*,
    const TensorImpl*) {
  TORCH_INTERNAL_ASSERT(
      0,
      "attempted to call `sym_sizes` on Tensor with nontrivial PyObject after corresponding interpreter died");
}

static c10::Layout noop_layout_fn(const PyInterpreter*, const TensorImpl*) {
  TORCH_INTERNAL_ASSERT(
      0,
      "attempted to call `layout` on Tensor with nontrivial PyObject after corresponding interpreter died");
}

static c10::SymInt noop_sym_numel_fn(
    const PyInterpreter*,
    const TensorImpl*) {
  TORCH_INTERNAL_ASSERT(
      0,
      "attempted to call `sym_numel` on Tensor with nontrivial PyObject after corresponding interpreter died");
}

void PyInterpreter::disarm() noexcept {
  name_fn_ = &noop_name_fn;
  decref_fn_ = &noop_decref_fn;
  detach_fn_ = &noop_detach_fn;
  dispatch_fn_ = &noop_dispatch_fn;
  is_contiguous_fn_ = &noop_is_contiguous_fn;
  device_fn_ = &noop_device_fn;
  dim_fn_ = &noop_dim_fn;
  strides_fn_ = &noop_strides_fn;
  sizes_fn_ = &noop_sizes_fn;
  sym_sizes_fn_ = &noop_sym_sizes_fn;
  layout_fn_ = &noop_layout_fn;
  sym_numel_fn_ = &noop_sym_numel_fn;
}

} // namespace impl
} // namespace c10
