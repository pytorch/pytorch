#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <sstream>

namespace c10 {

// This a "fake" kernel which doesn't actually do anything.  Instead, it is a
// distinguished kernel which is special cased by the dispatch table to
// be handled specially.  Its semantics is that it redispatches to the
// *next* dispatch key that would have been processed, skipping the current
// one.
void fallthrough_kernel(OperatorKernel*, const OperatorHandle&, DispatchKeySet, Stack*) {
  TORCH_INTERNAL_ASSERT(0,
    "fallthrough_kernel was executed but it should have been short-circuited by the dispatcher. "
    "This could occur if you registered a fallthrough kernel as a override for a specific operator "
    "(as opposed to a backend fallback); this is NOT currently supported, and we do not intend to "
    "add support for it in the near future.  If you do find yourself in need of this, "
    "let us know in the bug tracker.");
}

void ambiguous_autogradother_kernel(OperatorKernel*, const OperatorHandle& op, DispatchKeySet, Stack*) {
  TORCH_INTERNAL_ASSERT(0,
    op.operator_name(), " has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. "
    "This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering "
    "(see Note [Ambiguity in AutogradOther kernel]). "
    "If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated "
    "Autograd dispatch key for the backend.\n",
    "If you only want to run inference instead of training, add `c10::InferenceMode mode;` "
    "before model.forward(). Note this guard is only available in C++ but not Python at present.",
    "\nCanonical state\n~~~~~~~~~~~\n", op.dumpState(), "\n\n");
}

void named_not_supported_kernel(OperatorKernel*, const OperatorHandle& op, DispatchKeySet, Stack*) {
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
  auto boxed_kernel_fn = boxed_kernel_func_.getFnPtr();
  if (boxed_kernel_fn == fallthrough_kernel) {
    oss << "fallthrough ";
  }
  if (boxed_kernel_fn) {
    oss << "boxed ";
  }
  if (unboxed_kernel_func_) {
    oss << "unboxed ";
  }
  return oss.str();
}

bool KernelFunction::_equalsBoxedAndUnboxed(const KernelFunction& other) const {
  return boxed_kernel_func_.getFnPtr() == other.boxed_kernel_func_.getFnPtr() &&
         unboxed_kernel_func_ == other.unboxed_kernel_func_;
}

} // namespace c10
