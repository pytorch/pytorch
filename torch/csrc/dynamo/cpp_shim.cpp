#include <torch/csrc/dynamo/cpp_shim.h>

#include <ATen/record_function.h>

struct _PytorchRecordFunctionState {
  at::RecordFunction guard;

  explicit _PytorchRecordFunctionState(at::StepCallbacks&& callbacks)
      : guard(std::move(callbacks)) {}
};

// Returns nullptr when nothing would record the call, which is every call
// while profiling is off, so that path allocates nothing.
_PytorchRecordFunctionState* _pytorch_record_function_enter(const char* name) {
  auto callbacks = at::getStepCallbacksUnlessEmpty(at::RecordScope::FUNCTION);
  if (!callbacks) {
    return nullptr;
  }
  auto* state = new _PytorchRecordFunctionState(std::move(*callbacks));
  state->guard.before(name);
  return state;
}

void _pytorch_record_function_exit(_PytorchRecordFunctionState* state) {
  if (state == nullptr) {
    return;
  }
  delete state;
}
