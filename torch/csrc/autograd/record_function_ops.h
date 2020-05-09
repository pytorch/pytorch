#pragma once
#include <ATen/record_function.h>
#include <torch/csrc/utils/future.h>

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
template <typename T>
void _call_end_callbacks_on_fut(
    const at::Tensor& handle,
    const std::shared_ptr<torch::utils::Future<T>> fut) {
  // Save and pass thread local state into the callback
  at::ThreadLocalState tls_state;
  // Add a callback onto the future to mark run RecordFunction's end callbacks
  // when the future is completed.
  fut->addCallback(
      // Copy handle and tls_state by value to persist after the python
      // context manager is exited.
      [handle, tls_state = std::move(tls_state)]() {
        TORCH_INTERNAL_ASSERT(
            handle.defined(),
            "Undefined RecordFunction handle. This can happen if the handle is "
            "not correctly persisted and is destroyed before the future is "
            "realized.");
        at::ThreadLocalStateGuard g(tls_state);
        auto& rec = getRecordFunctionFromTensor(handle);
        rec._end();
      });
}


} // namespace profiler
} // namespace autograd
} // namespace torch
