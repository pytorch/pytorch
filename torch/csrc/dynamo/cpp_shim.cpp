#include <ATen/record_function.h>
#include <torch/csrc/dynamo/cpp_shim.h>

struct _PytorchRecordFunctionState {
  at::RecordFunction guard;

  _PytorchRecordFunctionState() : guard(at::RecordScope::FUNCTION) {}
};

_PytorchRecordFunctionState* _pytorch_record_function_enter(const char* name) {
  _PytorchRecordFunctionState* state = new _PytorchRecordFunctionState();
  state->guard.before(name);
  return state;
}

static inline _PytorchRecordFunctionState*
_pytorch_record_function_enter_with_kwinputs(
    const char* name,
    const std::unordered_map<std::string, c10::IValue>* kwargs) {
  _PytorchRecordFunctionState* state = new _PytorchRecordFunctionState();
  std::vector<c10::IValue> args;
  state->guard.before(name, &args, kwargs);
  return state;
}

_PytorchRecordFunctionState* _pytorch_record_function_enter_with_context(
    const char* name,
    const char* context) {
  auto map = std::unordered_map<std::string, c10::IValue>();
  map.insert({"context", c10::IValue(context)});
  return _pytorch_record_function_enter_with_kwinputs(name, &map);
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
