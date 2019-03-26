#pragma once

#include <ATen/core/ivalue.h>
#include <c10/util/SmallVector.h>

namespace torch { namespace autograd {

struct Function;

namespace profiler {

using GetPackedInputsCallback = std::function<std::vector<c10::IValue>()>;

struct FunctionCallContext {
  explicit FunctionCallContext(
      Function* fn, GetPackedInputsCallback cb = nullptr);
  explicit FunctionCallContext(
      std::string name, int64_t sequence_nr = -1, GetPackedInputsCallback cb = nullptr);
  explicit FunctionCallContext(
      const char* name_ptr, int64_t sequence_nr = -1, GetPackedInputsCallback cb = nullptr);

  inline Function* func() const {
    return fn_;
  }

  inline const char* name() const {
    return name_ptr_;
  }

  inline int64_t seqNr() const {
    return sequence_nr_;
  }

  inline bool hasOwnedName() const {
    return owned_name_ != nullptr;
  }

  const std::vector<c10::IValue>& inputs() const {
    if (inputs_cb_ && !inputs_initialized_) {
      inputs_ = inputs_cb_();
      inputs_initialized_ = true;
    }
    return inputs_;
  }

  inline const std::shared_ptr<FunctionCallContext>& parent() const {
    return parent_ctx_;
  }

  inline void setParent(const std::shared_ptr<FunctionCallContext>& parent) {
    parent_ctx_ = parent;
  }

 private:
  Function* fn_ = nullptr;
  std::unique_ptr<std::string> owned_name_;
  const char* name_ptr_ = nullptr;
  int64_t sequence_nr_ = -1;

  std::shared_ptr<FunctionCallContext> parent_ctx_;

  GetPackedInputsCallback inputs_cb_ = nullptr;
  mutable bool inputs_initialized_ = false;
  // initialized lazily by inputs_cb_
  mutable std::vector<c10::IValue> inputs_;
};

struct RecordFunction {
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
      GetPackedInputsCallback cb) : RecordFunction(name, -1, cb) {};

  explicit RecordFunction(
      const char* name,
      GetPackedInputsCallback cb) : RecordFunction(name, -1, cb) {};

  virtual ~RecordFunction();

 private:
  template<typename... Args>
  void processCallbacks(Args&&... args);

  std::shared_ptr<FunctionCallContext> ctx_;
};

// WARNING: all calls to pushCallback/popCallback are not thread safe and
// must not overlap with other code execution
using RecordFunctionCallback = std::function<void(const FunctionCallContext&)>;
void pushCallback(RecordFunctionCallback, RecordFunctionCallback);
void pushCallback(RecordFunctionCallback);
void popCallback();
// Functions to pass the context across fork calls
const std::shared_ptr<FunctionCallContext>& currentFunctionCallContext();
void setCurrentFunctionCallContext(const std::shared_ptr<FunctionCallContext>&);

} // namespace profiler
}} // namespace torch::autograd
