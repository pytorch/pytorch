#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace {

void pythonFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
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
        interpreter->dispatch(op, stack);
        return;
      }
    } else if (ivalue.isList()) {
      for (const auto& nv : ivalue.toListRef()) {
        if (nv.isTensor()) {
          auto* interpreter = nv.unsafeToTensorImpl()->pyobj_interpreter();
          if (interpreter) {
            interpreter->dispatch(op, stack);
            return;
          }
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
