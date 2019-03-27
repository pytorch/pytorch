#pragma once

#include <ATen/core/ivalue.h>
#include <c10/util/SmallVector.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch { namespace autograd {

struct Function;

namespace profiler {

struct TORCH_API StringView {
  StringView() : StringView(nullptr) {}
  explicit StringView(const char* str_ptr)
    : owned_str_ptr_(nullptr), str_ptr_(str_ptr) {}
  explicit StringView(std::string str)
    : owned_str_ptr_(std::make_shared<std::string>(std::move(str))),
      str_ptr_(owned_str_ptr_->c_str()) {}

  inline const char* str() const {
    return str_ptr_;
  }
 private:
  std::shared_ptr<std::string> owned_str_ptr_;
  const char* str_ptr_;
};

using GetPackedInputsCallback = std::function<std::vector<c10::IValue>()>;

struct TORCH_API RecordFunction {
  explicit RecordFunction(Function* fn, GetPackedInputsCallback cb = nullptr);

  explicit RecordFunction(
      std::string name,
      int64_t current_sequence_nr = -1,
      GetPackedInputsCallback cb = nullptr);

  explicit RecordFunction(
      const char* name,
      int64_t current_sequence_nr = -1,
      GetPackedInputsCallback cb = nullptr);

  explicit RecordFunction(
      std::string name,
      GetPackedInputsCallback cb) : RecordFunction(name, -1, cb) {}

  explicit RecordFunction(
      const char* name,
      GetPackedInputsCallback cb) : RecordFunction(name, -1, cb) {}

  virtual ~RecordFunction();


  inline Function* func() const {
    return fn_;
  }

  inline const StringView& name() const {
    return name_;
  }

  inline int64_t seqNr() const {
    return sequence_nr_;
  }

  const std::vector<c10::IValue>& inputs() const {
    if (inputs_cb_ && !inputs_initialized_) {
      inputs_ = inputs_cb_();
      inputs_initialized_ = true;
    }
    return inputs_;
  }

  inline const RecordFunction* parent() const {
    return parent_;
  }

 private:
  void processCallbacks();

  Function* fn_ = nullptr;
  StringView name_;
  int64_t sequence_nr_ = -1;

  RecordFunction* parent_ = nullptr;

  GetPackedInputsCallback inputs_cb_ = nullptr;
  mutable bool inputs_initialized_ = false;
  // initialized lazily by inputs_cb_
  mutable std::vector<c10::IValue> inputs_;
};

// WARNING: all calls to pushCallback/popCallback are not thread safe and
// must not overlap with other code execution
using RecordFunctionCallback = std::function<void(const RecordFunction&)>;
TORCH_API void pushCallback(RecordFunctionCallback, RecordFunctionCallback);
TORCH_API void pushCallback(RecordFunctionCallback);
TORCH_API void popCallback();

} // namespace profiler
}} // namespace torch::autograd
