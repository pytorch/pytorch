#include <torch/csrc/dynamo/cpp_shim.h>

#include <ATen/record_function.h>

_PytorchRecordFunctionState _pytorch_record_function_enter(const char* name) {
  at::RecordFunction* guard = new at::RecordFunction(at::RecordScope::FUNCTION);
  guard->before(name);
  return {static_cast<void*>(guard)};
}

void _pytorch_record_function_exit(_PytorchRecordFunctionState* state) {
  at::RecordFunction* guard = static_cast<at::RecordFunction*>(state->guard);
  delete guard;
  state->guard = nullptr;
}
