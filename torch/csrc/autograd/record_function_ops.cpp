#include <torch/csrc/autograd/record_function_ops.h>
#include <ATen/cpp_custom_type_hack.h>
#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/distributed/rpc/message.h>

namespace caffe2 {
// Required for cpp_custom_type_hack to work
// NOLINTNEXTLINE(bugprone-exception-escape)
CAFFE_KNOWN_TYPE(torch::autograd::profiler::RecordFunction);
} // namespace caffe2

namespace torch {
namespace autograd {
namespace profiler {

at::Tensor record_function_enter(const std::string& name) {
  auto rec = std::make_unique<RecordFunction>(RecordScope::USER_SCOPE);
  // Only add new scope if profiling is enabled.
  if (auto* current = rec->current()) {
    AT_ASSERT(
        current->name() == StringView("profiler::_record_function_enter"));
    // RecordFunction requires parent_ to be alive for it's entire lifetime.
    // Since the currently active RecordFunction will only live for the lifetime
    // of this op we need to end it early so the new RecordFunction we create is
    // a direct child of the parent RecordFunction.
    current->_end();
    rec->_before(name);
  }
  return at::cpp_custom_type_hack::create(std::move(rec), at::TensorOptions());
}

RecordFunction& getRecordFunctionFromTensor(const at::Tensor& handle) {
  auto& rec = at::cpp_custom_type_hack::cast<RecordFunction>(handle);
  return rec;
}

void record_function_exit(const at::Tensor& handle) {
  // We don't actually need to do anything with handle just need to persist the
  // lifetime until now.
  auto& rec = getRecordFunctionFromTensor(handle);
  if (auto* current = RecordFunction::current()) {
    AT_ASSERT(current->name() == StringView("profiler::_record_function_exit"));
    current->_end();
  }
  rec._end();
}

template <typename T>
void _call_end_callbacks_on_fut(
    const at::Tensor& handle,
    const std::shared_ptr<torch::utils::Future<T>> fut) {
  // Add a callback onto the future to mark run RecordFunction's end callbacks
  // when the future is completed.
  fut->addCallback(
      // Copy handle by value to persist after the python context manager is
      // exited.
      [handle](
          const T& /* unused */,
          const c10::optional<torch::utils::FutureError>& /* unused */) {
        TORCH_INTERNAL_ASSERT(
            handle.defined(),
            "Undefined RecordFunction handle. This can happen if the handle is "
            "not correctly persisted and is destroyed before the future is "
            "realized.");
        auto& rec = getRecordFunctionFromTensor(handle);
        rec._end();
      });
}
// Explicit template instantiation of _call_end_callbacks_on_fut.
template void _call_end_callbacks_on_fut(
    const at::Tensor& handle,
    const std::shared_ptr<torch::distributed::rpc::FutureMessage>);

// Internal only, do not use directly, use Python's record_function()
static auto registry =
    RegisterOperators()
        .op("profiler::_record_function_enter", &record_function_enter)
        .op("profiler::_record_function_exit", &record_function_exit);

} // namespace profiler
} // namespace autograd
} // namespace torch
