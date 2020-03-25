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

  friend std::ostream& operator<<(std::ostream& os, const StringView& dt) {
    os << dt.str();
    return os;
  }

  friend bool operator==(const StringView& lhs, const StringView& rhs) {
    return strcmp(lhs.str(), rhs.str()) == 0;
  }

  friend bool operator!=(const StringView& lhs, const StringView& rhs) {
    return !(lhs == rhs);
  }

 private:
  std::shared_ptr<std::string> owned_str_ptr_;
  const char* str_ptr_;
};

struct TORCH_API RecordFunction {
  // Default constructor is used with before function called afterwards
  RecordFunction() {}

  RecordFunction(const RecordFunction&) = delete;
  RecordFunction& operator=(const RecordFunction&) = delete;

  // current returns the currently active RecordFunction in this thread.
  static RecordFunction* current();

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

  bool active() const {
    return initialized_;
  }

  // Internal, only for the use within RECORD_FUNCTION macro;
  // enables this record function to run sampled callbacks
  void _setRunSampled(bool run_sampled) {
    run_sampled_ = run_sampled;
  }

  // Internal, only for the use within RECORD_FUNCTION macro;
  // sets this function as the current() thread local function;
  // original value of current() is restored in destructor
  void _setCurrent();

  // Executes end callbacks
  void end();

  // Retrieves the thread_id that this RecordFunction ran start callbacks with.
  // Useful for writing thread safe end callbacks that may be potentially
  // executed in a different thread (async ops)
  inline uint16_t getStartCallbacksThreadId() const {
    return threadId_;
  }

  // Get logical thread_id for the current thread
  static uint16_t getCurrentThreadId();

 private:
  void processCallbacks();

  Node* fn_ = nullptr;
  StringView name_;
  int64_t sequence_nr_ = -1;
  std::vector<c10::IValue> inputs_;
  // parent_ points to the parent RecordFunction and must out live this;
  // only to be used together with RECORD_FUNCTION macro
  RecordFunction* parent_ = nullptr;

  bool initialized_ = false;
  bool run_sampled_ = false;

  // is_current_ true means that this record function updates thread local
  // current record function pointer;
  // true only in case of scope-based record functions, i.e.
  // RECORD_FUNCTION macro
  bool is_current_ = false;

  // The logical thread_id that this RecordFunction was created with.
  uint16_t threadId_ = 0;
};

TORCH_API bool hasCallbacks();
TORCH_API bool needsInputs();
TORCH_API bool hasNonSampledCallbacks();

TORCH_API void setSamplingProbability(double);
TORCH_API double getSamplingProbability();

TORCH_API bool shouldRunSampledCallbacks();
// Given a record function, run the (possibly sampled) start callbacks that have
// been pushed via pushCallback().
TORCH_API void runBeforeCallbacks(
    RecordFunction* rf,
    const std::string& funcName);

// optional argument - function's seq_no
#define RECORD_FUNCTION(fn, inputs, ...) \
  torch::autograd::profiler::RecordFunction guard; \
  if (torch::autograd::profiler::hasCallbacks()) { \
    auto run_sampled = torch::autograd::profiler::shouldRunSampledCallbacks(); \
    if (run_sampled || torch::autograd::profiler::hasNonSampledCallbacks()) { \
      guard._setCurrent(); \
      guard._setRunSampled(run_sampled); \
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
