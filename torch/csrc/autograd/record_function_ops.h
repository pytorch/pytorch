#pragma once
#include <ATen/record_function.h>

namespace torch {
namespace autograd {
namespace profiler {
// Creates a new profiling scope using RecordFunction and invokes its starting
// callbacks.
TORCH_API at::Tensor record_function_enter(const std::string& name);

// Schedules RecordFunction's end callbacks to be run on completion of a future.
TORCH_API c10::intrusive_ptr<c10::ivalue::Future> _call_end_callbacks_on_fut(
    const at::Tensor& handle,
    const c10::intrusive_ptr<c10::ivalue::Future>& fut);

} // namespace profiler
} // namespace autograd
} // namespace torch
