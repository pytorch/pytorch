#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <sstream>

namespace c10 {

// This a "fake" kernel which doesn't actually do anything.  Instead, it is a
// distinguished kernel which is special cased by the dispatch table to
// be handled specially.  Its semantics is that it redispatches to the
// *next* dispatch key that would have been processed, skipping the current
// one.
void fallthrough_kernel(OperatorKernel*, const OperatorHandle&, Stack*) {
  TORCH_INTERNAL_ASSERT(0,
    "fallthrough_kernel was executed but it should have been short-circuited by the dispatcher. "
    "This could occur if you registered a fallthrough kernel as a override for a specific operator "
    "(as opposed to a backend fallback); this is NOT currently supported, and we do not intend to "
    "add support for it in the near future.  If you do find yourself in need of this, "
    "let us know in the bug tracker.");
}

void ambiguous_autogradother_kernel(OperatorKernel*, const OperatorHandle& op, Stack*) {
  TORCH_INTERNAL_ASSERT(0,
    op.operator_name(), " has kernels registered to both Math and a backend mapped to AutogradOther. "
    "This makes the backend kernel unreachable (see Note [Ambiguity in AutogradOther kernel]). "
    "If it's intended to override Math kernel behavior, please open an issue to request a dedicated "
    "Autograd dispatch key for the backend.");
}

void named_not_supported_kernel(OperatorKernel*, const OperatorHandle& op, Stack*) {
  // DO NOT LOOK AT STACK, YOU HAVE SHORT CIRCUITED BOXING
  // See Note [named_not_supported_kernel]
  TORCH_CHECK(0,
    op.operator_name(), " is not yet supported with named tensors. Please drop names via "
    "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
    "and set names on the result of the operation."
    );
}

// single line summary of state
std::string KernelFunction::dumpState() const {
  std::ostringstream oss;
  if (boxed_kernel_func_ == fallthrough_kernel) {
    oss << "fallthrough ";
  }
  if (boxed_kernel_func_) {
    oss << "boxed ";
  }
  if (unboxed_kernel_func_) {
    oss << "unboxed ";
  }
  return oss.str();
}

bool KernelFunction::_equalsBoxedAndUnboxed(const KernelFunction& other) const {
  return boxed_kernel_func_ == other.boxed_kernel_func_ &&
         unboxed_kernel_func_ == other.unboxed_kernel_func_;
}

void KernelFunction::checkBoxedKernel(const OperatorHandle& opHandle) const {
  if (C10_UNLIKELY(boxed_kernel_func_ == nullptr)) {
    if (unboxed_kernel_func_ == nullptr) {
      TORCH_INTERNAL_ASSERT(
          false,
          "Tried to call KernelFunction::callBoxed() on an uninitialized KernelFunction.",
          " opname: ",
          opHandle.operator_name(),
          " If you're using mobile selective build please make sure to include all ops exported from `torch.jit.export_opnames(model)`.");
    } else {
      // TODO We want to introduce the invariant that all kernels must be callable in a boxed way, then this case should be impossible.
      TORCH_INTERNAL_ASSERT(
          false,
          "Tried to call KernelFunction::callBoxed() on a KernelFunction that can only be called with KernelFunction::call().",
          " opname: ",
          opHandle.operator_name(),
          " If you're using mobile selective build please make sure to include all ops exported from `torch.jit.export_opnames(model)`.");
    }
  }
}

} // namespace c10
