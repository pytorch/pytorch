#pragma once

#include <ATen/core/ivalue.h>
#include <c10/util/SmallVector.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch { namespace autograd {

struct Node;

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

  // before function initializes RecordFunction members and calls
  // start callbacks
  void before(const char* name, int64_t sequence_nr = -1);
  void before(std::string name, int64_t sequence_nr = -1);
  void before(Node* fn, int64_t sequence_nr = -1);

  template<typename F>
  void before(
      F fn,
      c10::ArrayRef<c10::IValue> args,
      int64_t current_sequence_nr = -1) {
    inputs_ = args.vec();
    before(fn, current_sequence_nr);
  }

  template<typename F>
  void before(
      F fn,
      std::vector<c10::IValue>&& args,
      int64_t current_sequence_nr = -1) {
    inputs_ = std::move(args);
    before(fn, current_sequence_nr);
  }

  // Destructor calls end callbacks
  virtual ~RecordFunction();

  inline Node* func() const {
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

  void setRunSampled(bool run_sampled) {
    run_sampled_ = run_sampled;
  }

 private:
  void processCallbacks();

  Node* fn_ = nullptr;
  StringView name_;
  int64_t sequence_nr_ = -1;
  std::vector<c10::IValue> inputs_;
  RecordFunction* parent_ = nullptr;

  bool initialized_ = false;
  bool run_sampled_ = false;
};

TORCH_API bool hasCallbacks();
TORCH_API bool needsInputs();
TORCH_API bool hasNonSampledCallbacks();

TORCH_API void setSamplingProbability(double);
TORCH_API double getSamplingProbability();

TORCH_API bool shouldRunSampledCallbacks();

// optional argument - function's seq_no
#define RECORD_FUNCTION(fn, inputs, ...) \
  torch::autograd::profiler::RecordFunction guard; \
  if (torch::autograd::profiler::hasCallbacks()) { \
    auto run_sampled = torch::autograd::profiler::shouldRunSampledCallbacks(); \
    if (run_sampled || torch::autograd::profiler::hasNonSampledCallbacks()) { \
      guard.setRunSampled(run_sampled); \
      if (torch::autograd::profiler::needsInputs()) { \
        guard.before(fn, inputs, ##__VA_ARGS__); \
      } else { \
        guard.before(fn, ##__VA_ARGS__); \
      } \
    } \
  }

// WARNING: all calls to pushCallback/popCallback are not thread safe and
// must not overlap with other code execution
using RecordFunctionCallback = std::function<void(const RecordFunction&)>;
TORCH_API void pushCallback(
    RecordFunctionCallback start,
    RecordFunctionCallback end = [](const RecordFunction&){},
    bool needs_inputs = false,
    bool sampled = false);
TORCH_API void popCallback();

} // namespace profiler
}} // namespace torch::autograd
