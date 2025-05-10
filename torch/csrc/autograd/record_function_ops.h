#pragma once

#include <ATen/record_function.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <optional>

namespace torch::autograd::profiler {

struct PythonRecordFunction : public torch::CustomClassHolder {
  at::RecordFunction record;

  explicit PythonRecordFunction(
      at::RecordScope scope = at::RecordScope::FUNCTION)
      : record(scope) {}
};

// Creates a new profiling scope using RecordFunction and invokes its starting
// callbacks.
TORCH_API c10::intrusive_ptr<PythonRecordFunction> record_function_enter_new(
    const std::string& name,
    const std::optional<std::string>& args = std::nullopt);

// Schedules RecordFunction's end callbacks to be run on completion of a future.
TORCH_API c10::intrusive_ptr<c10::ivalue::Future> _call_end_callbacks_on_fut_new(
    const c10::intrusive_ptr<PythonRecordFunction>& record,
    const c10::intrusive_ptr<c10::ivalue::Future>& fut);

/**
 * RAII wrapper that behaves similarly to torch.profiler.record_function.
 */
class RecordFunction {
 public:
  RecordFunction() = delete;
  RecordFunction(const RecordFunction&) = default;
  RecordFunction& operator=(const RecordFunction&) = default;
  RecordFunction(RecordFunction&&) = default;
  RecordFunction& operator=(RecordFunction&&) = default;

  explicit RecordFunction(const std::string& name) {
    recordFunction_ =
        torch::autograd::profiler::record_function_enter_new(name);
  }

  ~RecordFunction() {
    recordFunction_->record.end();
  }

 private:
  c10::intrusive_ptr<torch::autograd::profiler::PythonRecordFunction>
      recordFunction_;
};

} // namespace torch::autograd::profiler
