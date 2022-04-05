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
    torch::jit::Stack* stack,
    const std::shared_ptr<SafePyObject>& type) {
  TORCH_INTERNAL_ASSERT(
      0,
      "attempted to dispatch (__torch_dispatch__) an operator on Tensor with nontrivial PyObject after corresponding interpreter died");
}

void PyInterpreter::disarm() noexcept {
  name_fn_ = &noop_name_fn;
  decref_fn_ = &noop_decref_fn;
  detach_fn_ = &noop_detach_fn;
  dispatch_fn_ = &noop_dispatch_fn;
}

} // namespace impl
} // namespace c10
