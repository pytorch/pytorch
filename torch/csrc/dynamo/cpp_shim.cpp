#include <torch/csrc/dynamo/cpp_shim.h>

#include <ATen/record_function.h>

struct _PytorchRecordFunctionState {
  at::RecordFunction guard;

  _PytorchRecordFunctionState() : guard(at::RecordScope::FUNCTION) {}
};

_PytorchRecordFunctionState* _pytorch_record_function_enter(const char* name) {
  _PytorchRecordFunctionState* state = new _PytorchRecordFunctionState();
  state->guard.before(name);
  return state;
}

void _pytorch_record_function_exit(_PytorchRecordFunctionState* state) {
  if (state == nullptr) {
    return;
  }
  delete state;
}

thread_local bool eval_frame_callback_enabled = true;

bool eval_frame_callback_enabled_get() {
  return eval_frame_callback_enabled;
}

void eval_frame_callback_enabled_set(bool enabled) {
  eval_frame_callback_enabled = enabled;
}
