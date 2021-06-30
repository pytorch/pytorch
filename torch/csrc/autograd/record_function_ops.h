#pragma once
#include <ATen/record_function.h>

#include <torch/csrc/jit/runtime/custom_operator.h>

namespace torch {
namespace autograd {
namespace profiler {

// Holder of RecordFunction, used to store the state of a RecordFunction
// object to record the enter and exit event for profiler.
struct RecordFunctionHolder : torch::CustomClassHolder {
  RecordFunctionHolder() {}
  void enter(const std::string& name);
  void exit();
private:
  std::unique_ptr<at::RecordFunction> record_function_;
};

// Creates a new profiling scope using RecordFunction and invokes its starting
// callbacks.
TORCH_API c10::intrusive_ptr<RecordFunctionHolder> record_function_enter(
    const std::string& name);

// Schedules RecordFunction's end callbacks to be run on completion of a future.
TORCH_API c10::intrusive_ptr<c10::ivalue::Future> _call_end_callbacks_on_fut(
    c10::intrusive_ptr<RecordFunctionHolder> holder,
    const c10::intrusive_ptr<c10::ivalue::Future>& fut);

} // namespace profiler
} // namespace autograd
} // namespace torch
