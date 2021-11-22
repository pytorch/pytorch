#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/PythonModeTLS.h>

namespace {

void pythonFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // If Python Mode is active, use its PyInterpreter for dispatch
  const auto& maybe_python_mode_state = at::impl::PythonModeTLS::get_state();
  if (maybe_python_mode_state) {
    maybe_python_mode_state->pyinterpreter()->dispatch(op, stack, maybe_python_mode_state);
    return;
  }

  // Otherwise, find a PyInterpreter on a Tensor
  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  // It is safe to dispatch on the very first Tensor with a pyobj_interpreter
  // without checking the interpreters of any of the arguments, because when
  // we actually run dispatch(), we will take out PyObjects in the context
  // of that interpreter, and this will ensure that everyone is on the same
  // interpreter.
  for (const auto& ivalue : torch::jit::last(*stack, num_arguments)) {
    if (ivalue.isTensor()) {
      auto* interpreter = ivalue.unsafeToTensorImpl()->pyobj_interpreter();
      if (interpreter) {
        interpreter->dispatch(op, stack, nullptr);
        return;
      }
    } else if (ivalue.isTensorList()) {
      // NB: use toListRef as it doesn't induce refcount bumps (toTensorListRef
      // is not a thing)
      for (const auto& nv : ivalue.toListRef()) {
        auto* interpreter = nv.unsafeToTensorImpl()->pyobj_interpreter();
        if (interpreter) {
          interpreter->dispatch(op, stack, nullptr);
          return;
        }
      }
    }
  }
  TORCH_INTERNAL_ASSERT(0, "Hit Python dispatch key but no arguments had PyInterpreter (no tensor args?)");
}

} // anonymous namespace

TORCH_LIBRARY_IMPL(_, Python, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&pythonFallback>());
}
