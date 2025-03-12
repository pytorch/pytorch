#pragma once

#include <torch/library.h> //@manual=//caffe2:libtorch
#include "torch/csrc/autograd/record_function_ops.h" //@manual=//caffe2:libtorch
namespace torch::nativert {

/**
 * RAII-style wrapper that behaves similarly to torch.profiler.record_function.
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

} // namespace torch::nativert
