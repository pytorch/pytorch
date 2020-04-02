#pragma once
#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/utils/future.h>

namespace torch {
namespace autograd {
namespace profiler {
// Creates a new profiling scope using RecordFunction and invokes its starting
// callbacks.
at::Tensor record_function_enter(const std::string& name);

// Cast Tensor that was created with at::cpp_custom_type_hack back to
// RecordFunction. This is a temporary workaround until RecordFunction is
// registered as a custom C++ class.
TORCH_API RecordFunction& getRecordFunctionFromTensor(const at::Tensor& handle);

// Schedules RecordFunction's end callbacks to be run on completion of a future.
template <typename T>
TORCH_API void _call_end_callbacks_on_fut(
    const at::Tensor& handle,
    const std::shared_ptr<torch::utils::Future<T>> fut);

// Ends the profiling scope created with record_function_enter.
void record_function_exit(const at::Tensor& handle);

} // namespace profiler
} // namespace autograd
} // namespace torch
