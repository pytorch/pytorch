#include <ATen/cpp_custom_type_hack.h>
#include <ATen/record_function.h>
#include <ATen/ThreadLocalState.h>

#include <torch/csrc/jit/runtime/custom_operator.h>

namespace caffe2 {
// Required for cpp_custom_type_hack to work
// NOLINTNEXTLINE(bugprone-exception-escape)
CAFFE_KNOWN_TYPE(at::RecordFunction);
} // namespace caffe2

namespace torch {
namespace autograd {
namespace profiler {

// Creates a new profiling scope using RecordFunction and invokes its starting
// callbacks.
at::Tensor record_function_enter(const std::string& name) {
  auto rec = std::make_unique<at::RecordFunction>(at::RecordScope::USER_SCOPE);
  if (auto* current = rec->current()) {
    if (current->name().str() == std::string("profiler::_record_function_enter")) {
      // RecordFunction requires parent_ to be alive for it's entire lifetime.
      // Since the currently active RecordFunction will only live for the lifetime
      // of this op we need to end it early so the new RecordFunction we create is
      // a direct child of the parent RecordFunction.
      current->_end();
    }
  }
  rec->_before(name);
  return at::cpp_custom_type_hack::create(std::move(rec), at::TensorOptions());
}

at::RecordFunction& getRecordFunctionFromTensor(const at::Tensor& handle) {
  auto& rec = at::cpp_custom_type_hack::cast<at::RecordFunction>(handle);
  return rec;
}

// Ends the profiling scope created with record_function_enter.
void record_function_exit(const at::Tensor& handle) {
  // We don't actually need to do anything with handle just need to persist the
  // lifetime until now.
  auto& rec = getRecordFunctionFromTensor(handle);
  if (auto* current = rec.current()) {
    if (current->name().str() == std::string("profiler::_record_function_exit")) {
      current->_end();
    }
  }
  rec._end();
}

void _call_end_callbacks_on_fut(
    const at::Tensor& handle,
    const c10::intrusive_ptr<c10::ivalue::Future>& fut) {
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

// Internal only, do not use directly, use Python's record_function()
static auto registry =
    RegisterOperators()
        .op("profiler::_record_function_enter", &record_function_enter)
        .op("profiler::_record_function_exit", &record_function_exit);

// Needed to register JIT operator in operator registry below
c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

jit::RegisterOperators reg_fut_ops({
    jit::Operator(
        "profiler::_call_end_callbacks_on_jit_fut(Tensor x, Future(t) y) -> ()",
        [](jit::Stack& stack) {
          // Pop inputs, which should be a future and a tensor
          auto fut = jit::pop(stack).toFuture();
          auto tensor = jit::pop(stack).toTensor();
          _call_end_callbacks_on_fut(tensor, fut);
          return 0;
        },
        aliasAnalysisFromSchema()),
});

} // namespace profiler
} // namespace autograd
} // namespace torch
