#include <torch/csrc/autograd/record_function_ops.h>
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

c10::intrusive_ptr<RecordFunction> record_function_enter(
    const std::string& name) {
  auto rec = c10::make_intrusive<RecordFunction>();
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
  return rec;
}

RecordFunction& getRecordFunctionFromTensor(const at::Tensor& handle) {
  auto& rec = at::cpp_custom_type_hack::cast<RecordFunction>(handle);
  return rec;
}

void record_function_exit(const c10::intrusive_ptr<RecordFunction>& instance) {
  // End the current RecordFunction, which should be
  // profiler::_record_function_exit to ensure the creating scope outlives it.
  if (auto* current = RecordFunction::current()) {
    AT_ASSERT(current->name() == StringView("profiler::_record_function_exit"));
    current->end();
  }
  if (instance->active()) {
    instance->end();
  }
}

// The following will bind the class to TorchScript with the qualified name
// torch.classes.profiler.RecordFunction. Note that this class is only meant to
// be used in conjunction with the record_function python context manager.
static auto torchScriptRecordFunction =
    torch::class_<RecordFunction>("profiler", "RecordFunction")
        .def(torch::init<>());

static auto registry =
    RegisterOperators()
        .op(RegisterOperators()
                .options()
                .schema(
                    "profiler::_record_function_enter(str x) -> __torch__.torch.classes.profiler.RecordFunction y")
                .catchAllKernel<
                    decltype(record_function_enter),
                    &record_function_enter>())
        .op(RegisterOperators()
                .options()
                .schema(
                    "profiler::_record_function_exit(__torch__.torch.classes.profiler.RecordFunction x) -> ()")
                .catchAllKernel<
                    decltype(record_function_exit),
                    &record_function_exit>());

} // namespace profiler
} // namespace autograd
} // namespace torch
