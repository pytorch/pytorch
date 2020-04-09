#pragma once

#include <ATen/core/ivalue.h>
#include <c10/util/SmallVector.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <functional>

namespace torch { namespace autograd {

struct Node;

namespace profiler {

// Kind of record function scope;
// workaround for the older GCC versions:
#ifndef _MSC_VER
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wattributes"
#endif
enum class TORCH_API RecordScope : uint8_t {
  // c10/ATen ops, autograd nodes
  FUNCTION = 0,
  // TorchScript functions, methods
  TORCHSCRIPT_FUNCTION,
  // User defined scope (e.g. with record_function())
  USER_SCOPE,
  NUM_SCOPES, // must be the last in the list
};
#ifndef _MSC_VER
#  pragma GCC diagnostic pop
#endif

} // namespace profiler
} // namespace autograd
} // namespace torch

namespace std {
template <>
struct hash<torch::autograd::profiler::RecordScope> {
  inline size_t operator()(
      const torch::autograd::profiler::RecordScope& sc) const {
    return static_cast<std::size_t>(sc);
  }
};
} // namespace std

namespace torch {
namespace autograd {
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

// Soft limit on the number of callbacks to use;
constexpr std::size_t kSoftLimitCallbacks = 32;

struct TORCH_API RecordFunction {
  // Default constructor is used with before function called afterwards:
  //  scope - record scope that this function tracks
  RecordFunction(
      RecordScope scope = RecordScope::FUNCTION);

  // Destructor calls end callbacks
  virtual ~RecordFunction();

  RecordFunction(const RecordFunction&) = delete;
  RecordFunction& operator=(const RecordFunction&) = delete;

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

  // Retrieves the thread_id that this RecordFunction ran start callbacks with.
  // Useful for writing thread safe end callbacks that may be potentially
  // executed in a different thread (async ops)
  inline uint16_t getStartCallbacksThreadId() const {
    return thread_id_;
  }

  inline RecordScope scope() const {
    return scope_;
  }

  // Current returns the currently active RecordFunction in this thread.
  static RecordFunction* current();

  // Returns logical thread_id for the current thread
  static uint16_t currentThreadId();

  // Internal functions, do not use directly;
  // might be called from python's context manager

  // Returns whether this record function runs callbacks
  bool _active() const {
    return active_;
  }

  // _before functions initialize RecordFunction members and call
  // start callbacks
  void _before(const char* name, int64_t sequence_nr = -1);
  void _before(std::string name, int64_t sequence_nr = -1);
  void _before(Node* fn, int64_t sequence_nr = -1);

  template<typename F>
  void _before(
      F fn,
      c10::ArrayRef<c10::IValue> args,
      int64_t current_sequence_nr = -1) {
    inputs_ = args.vec();
    _before(fn, current_sequence_nr);
  }

  template<typename F>
  void _before(
      F fn,
      std::vector<c10::IValue>&& args,
      int64_t current_sequence_nr = -1) {
    inputs_ = std::move(args);
    _before(fn, current_sequence_nr);
  }

  // Internal, only for the use within RECORD_FUNCTION macro;
  // sets this function as the current() thread local function;
  // original value of current() is restored in destructor/_end
  void _setCurrent();

  // Calls end callbacks
  void _end();

  // Returns whether some of the callbacks require function inputs
  static bool _needsInputs();

  inline uint64_t _callbacksVersion() const {
    return callbacks_version_;
  }

  inline void _setCallbacksVersion(uint64_t cv) {
    callbacks_version_ = cv;
  }

  // Returns boolean set of active (ran start callback) callbacks
  inline c10::SmallVector<bool, kSoftLimitCallbacks>& _activeCallbacks() {
    return active_callbacks_;
  }

 private:
  void processCallbacks();

  Node* fn_ = nullptr;
  StringView name_;
  int64_t sequence_nr_ = -1;
  std::vector<c10::IValue> inputs_;
  // parent_ points to the parent RecordFunction and must out live this;
  // only to be used together with RECORD_FUNCTION macro
  RecordFunction* parent_ = nullptr;

  // Holds the status of the callbacks after executing start callbacks.
  // If a start callback was not called (sampling) or returned false
  // (error or skipping the run), then the corresponding value in
  // the small vector is false and the end callback won't be called,
  // otherwise the value is true.
  c10::SmallVector<bool, kSoftLimitCallbacks> active_callbacks_;

  // is_current_ true means that this record function updates thread local
  // current record function pointer;
  // true only in case of scope-based record functions, i.e.
  // RECORD_FUNCTION macro
  bool is_current_ = false;
  bool active_ = false;
  const RecordScope scope_;

  // The logical thread_id that this RecordFunction was created with.
  uint16_t thread_id_ = 0;

  // Callbacks' version this record function was started with.
  // Used to ensure that the set of callbacks was not changed
  // during the record function's lifetime, between start and
  // end invocations.
  uint64_t callbacks_version_ = 0;
};

// Returns whether there're callbacks registered with pushCallback
TORCH_API bool hasCallbacks();

// Internal only, do not use:
// use C++ RECORD_* or python context manager record_function() instead;
// Given a record function, run the (possibly sampled) start callbacks that have
// been pushed via pushCallback().
TORCH_API void _runBeforeCallbacks(
    RecordFunction* rf,
    const std::string& funcName);

// Used in tests, overrides sampling probability for all callbacks;
TORCH_API void TEST_setGlobalSamplingProbability(double sampling_prob);
TORCH_API void TEST_unsetGlobalSamplingProbability();

// Using macro to minimize inputs copies,
// optional argument - function's seq_no
#define RECORD_FUNCTION_WITH_SCOPE(scope, fn, inputs, ...) \
  torch::autograd::profiler::RecordFunction guard(scope); \
  if (guard._active()) { \
    guard._setCurrent(); \
    if (torch::autograd::profiler::RecordFunction::_needsInputs()) { \
      guard._before(fn, inputs, ##__VA_ARGS__); \
    } else { \
      guard._before(fn, ##__VA_ARGS__); \
    } \
  }

#define RECORD_FUNCTION(fn, inputs, ...) \
  RECORD_FUNCTION_WITH_SCOPE( \
    torch::autograd::profiler::RecordScope::FUNCTION, \
    fn, inputs, ##__VA_ARGS__)

#define RECORD_TORCHSCRIPT_FUNCTION(mn, inputs) \
  RECORD_FUNCTION_WITH_SCOPE( \
    torch::autograd::profiler::RecordScope::TORCHSCRIPT_FUNCTION, mn, inputs)

// Custom user scopes in C++; similar to Python's 'with record_function("..."):'
#define RECORD_USER_SCOPE(fn) \
  RECORD_FUNCTION_WITH_SCOPE( \
    torch::autograd::profiler::RecordScope::USER_SCOPE, fn, {})

/**
 * pushCallback adds a pair of callbacks to run with RecordFunction:
 *  start, end - the callbacks to run when entering and exiting the scope;
 *    if start callback returns false, end callback won't be executed;
 *  needs_inputs - whether the callbacks need the inputs passed from the observed
 *    function/range; NOTE: passing the inputs incurs an additional overhead;
 *  sampling_prob - whether the callbacks are sampled and the sampling
 *    probability;
 *  scopes - types of scopes to execute the callbacks on (see RecordScope);
 *    passing empty set means the callbacks will be executed for all possible
 *    scope types
 *
 * WARNING: not thread safe, must not overlap with other PyTorch code execution
 */
TORCH_API void pushCallback(
    std::function<bool(const RecordFunction&)> start,
    std::function<void(const RecordFunction&)> end =
        [](const RecordFunction&) {},
    bool needs_inputs = false,
    double sampling_prob = 1.0,
    std::unordered_set<RecordScope, std::hash<RecordScope>> scopes =
        std::unordered_set<RecordScope, std::hash<RecordScope>>());

/**
 * popCallback removes the last pair of callbacks previously added with
 *  pushCallback
 *
 * WARNING: not thread safe, must not overlap with other PyTorch code execution
 */
TORCH_API void popCallback();

} // namespace profiler
}} // namespace torch::autograd
