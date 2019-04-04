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

struct TORCH_API RecordFunction {
  // Default constructor is used with before function called afterwards
  RecordFunction() {}

  // Constructors with parameters initialize RecordFunction
  // members and call start callbacks
  explicit RecordFunction(Function* fn);

  explicit RecordFunction(
      std::string name,
      int64_t current_sequence_nr = -1);

  explicit RecordFunction(
      const char* name,
      int64_t current_sequence_nr = -1);

  // before function initializes RecordFunction members and calls
  // start callbacks;
  // before function can be used only when RecordFunction is
  // constructed with default constructor RecordFunction::RecordFunction()
  void before(
      Function* fn,
      c10::optional<at::ArrayRef<c10::IValue>> inputs = c10::nullopt);

  void before(
      std::string name,
      c10::optional<at::ArrayRef<c10::IValue>> inputs = c10::nullopt,
      int64_t current_sequence_nr = -1);

  void before(
      const char* name,
      c10::optional<at::ArrayRef<c10::IValue>> inputs = c10::nullopt,
      int64_t current_sequence_nr = -1);

  // Destructor calls end callbacks
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
    return inputs_;
  }

  inline const RecordFunction* parent() const {
    return parent_;
  }

 private:
  void processCallbacks();
  void copyInputs(const at::ArrayRef<c10::IValue>& values) {
    inputs_ = values.vec();
  }

  Function* fn_ = nullptr;
  StringView name_;
  int64_t sequence_nr_ = -1;

  RecordFunction* parent_ = nullptr;

  std::vector<c10::IValue> inputs_;
  bool initialized_ = false;
};

#define _RECORD_FUNCTION(fn, inputs_case) \
  torch::autograd::profiler::RecordFunction guard; \
  switch (torch::autograd::profiler::recordFunctionState()) { \
    case torch::autograd::profiler::RecordFunctionState::NAME_ONLY: { \
      guard.before(fn); \
      break; \
    } \
    case torch::autograd::profiler::RecordFunctionState::NAME_AND_INPUTS: { \
      inputs_case \
      break; \
    } \
    case torch::autograd::profiler::RecordFunctionState::NO_CALLBACKS: { \
      break; \
    } \
  }

#define RECORD_FUNCTION_WITH_STACK(fn, stack_ref) \
  _RECORD_FUNCTION(fn, guard.before(fn, stack_ref);)

#define RECORD_FUNCTION_WITH_INPUTS_SEQ(fn, seq, ...) \
  _RECORD_FUNCTION(fn, \
    torch::jit::Stack stack({__VA_ARGS__}); \
    guard.before(fn, torch::jit::last(stack, stack.size()), seq); \
  )

#define RECORD_FUNCTION_WITH_INPUTS(fn, ...) \
  RECORD_FUNCTION_WITH_INPUTS_SEQ(fn, -1, __VA_ARGS__)

#define RECORD_FUNCTION_WITH_INPUT_RANGE(fn, start, end) \
  _RECORD_FUNCTION(fn, \
    torch::jit::Stack stack(start, end); \
    guard.before(fn, torch::jit::last(stack, stack.size())); \
  )

// WARNING: all calls to pushCallback/popCallback are not thread safe and
// must not overlap with other code execution
using RecordFunctionCallback = std::function<void(const RecordFunction&)>;
TORCH_API void pushCallback(
    RecordFunctionCallback start,
    RecordFunctionCallback end = [](const RecordFunction&){},
    bool needs_inputs = false);
TORCH_API void popCallback();

enum class RecordFunctionState {
  NO_CALLBACKS, // no callbacks registered, no need to use RecordFunction
  NAME_ONLY, // callbacks need only basic information (function name)
  NAME_AND_INPUTS, // callbacks need both name and function inputs
};
TORCH_API RecordFunctionState recordFunctionState();

} // namespace profiler
}} // namespace torch::autograd
