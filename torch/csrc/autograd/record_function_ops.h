#pragma once
#include <ATen/core/ivalue.h>
#include <ATen/record_function.h>

#include <ATen/ThreadLocalState.h>

namespace torch {
namespace autograd {
namespace profiler {

// Cast Tensor that was created with at::cpp_custom_type_hack back to
// RecordFunction. This is a temporary workaround until RecordFunction is
// registered as a custom C++ class
// (https://github.com/pytorch/pytorch/issues/35026).
TORCH_API at::RecordFunction& getRecordFunctionFromTensor(const at::Tensor& handle);

// Schedules RecordFunction's end callbacks to be run on completion of a future.
TORCH_API void _call_end_callbacks_on_fut(
    const at::Tensor& handle,
    const c10::intrusive_ptr<c10::ivalue::Future>& fut);

} // namespace profiler
} // namespace autograd
} // namespace torch
