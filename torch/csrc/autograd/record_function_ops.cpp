#include <ATen/cpp_custom_type_hack.h>
#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

namespace caffe2 {
// Required for cpp_custom_type_hack to work
// NOLINTNEXTLINE(bugprone-exception-escape)
CAFFE_KNOWN_TYPE(torch::autograd::profiler::RecordFunction);
} // namespace caffe2

namespace torch {
namespace autograd {
namespace profiler {

at::Tensor record_function_enter(const std::string& name) {
  auto rec = std::make_unique<RecordFunction>();
  // Only add new scope if profiling is enabled.
  if (auto* current = RecordFunction::current()) {
    AT_ASSERT(
        current->name() == StringView("profiler::_record_function_enter"));
    // RecordFunction requires parent_ to be alive for it's entire lifetime.
    // Since the currently active RecordFunction will only live for the lifetime
    // of this op we need to end it early so the new RecordFunction we create is
    // a direct child of the parent RecordFunction.
    current->end();

    runBeforeCallbacks(rec.get(), name);
  }
  return at::cpp_custom_type_hack::create(std::move(rec), at::TensorOptions());
}

void record_function_exit(const at::Tensor& handle) {
  // We don't actually need to do anything with handle just need to persist the
  // lifetime until now.
  auto& rec = at::cpp_custom_type_hack::cast<RecordFunction>(handle);
  if (auto* current = RecordFunction::current()) {
    AT_ASSERT(
        current->name() == StringView("profiler::_record_function_exit"));
  }
  if (rec.active()) {
    rec.end();
  }
}

static auto registry =
    RegisterOperators()
        .op("profiler::_record_function_enter", &record_function_enter)
        .op("profiler::_record_function_exit", &record_function_exit);

} // namespace profiler
} // namespace autograd
} // namespace torch
