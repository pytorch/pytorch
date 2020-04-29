#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/ThreadLocalState.h>
#include <c10/util/SmallVector.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/utils/memory.h>

#include <functional>
#include <random>

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

typedef c10::SmallVector<uint64_t, kSoftLimitCallbacks> handles_vector;

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
  // used in python's context manager

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

  // Internal, only for the use within RECORD_FUNCTION macro
  // (i.e. stack based RecordFunctions with scope lifetime);
  // sets this function as the current() thread local function;
  // original value of current() is restored in destructor/_end
  void _setCurrent();

  // Calls end callbacks
  void _end();

  // Returns whether some of the callbacks require function inputs
  bool _needsInputs();

  // Used internally to keep track of thread local and global callbacks
  // that were picked to run; must be sorted;
  // public because of anonymous "friend" class
  handles_vector sorted_active_tls_handles_;
  handles_vector sorted_active_global_handles_;
  // Whether this RecordFunction runs any callbacks
  bool active_ = false;
  /// Whether any of the picked callbacks require inputs
  bool needs_inputs_ = false;

 private:
  Node* fn_ = nullptr;
  StringView name_;
  int64_t sequence_nr_ = -1;
  std::vector<c10::IValue> inputs_;

  // parent_ points to the parent RecordFunction and must out live this;
  // only to be used together with RECORD_FUNCTION macro
  // (with stack based RecordFunction instances with scope lifetime)
  RecordFunction* parent_ = nullptr;

  // is_current_ true means that this record function updates thread local
  // current record function pointer;
  // true only in case of scope-based record functions, i.e.
  // RECORD_FUNCTION macro
  bool is_current_ = false;

  // Kind of scope this RecordFunction is observing
  const RecordScope scope_;

  // The logical thread_id that this RecordFunction was created with
  uint16_t thread_id_ = 0;
};

//
// PyTorch callbacks/observers API:
//

/**
 * RecordFunctionCallback represents a pair of callbacks to be used with
 * RecordFunction, members:
 *   start, end - the callbacks to run when entering and exiting the scope;
 *   needs_inputs - whether the callbacks need the inputs passed from the observed
 *     function/range; NOTE: passing the inputs incurs an additional overhead;
 *   sampling_probability - if not 1.0, then the callback is probabilistically sampled
 *     to run; NOTE: start and end callbacks always run as a pair and are sampled
 *     together;
 *   scopes - types of scopes to execute the callbacks on (see RecordScope);
 *     passing empty set means the callbacks will be executed for all possible
 *     scope types
 *   should_run - optional function that returns whether this callback should run;
 *     overwrites the effect of setting sampling_probability
 */
class TORCH_API RecordFunctionCallback {
 public:
  explicit RecordFunctionCallback(
      std::function<void(const RecordFunction&)> start,
      std::function<void(const RecordFunction&)> end =
        [](const RecordFunction&) {}):
      start_(std::move(start)),
      end_(std::move(end)) {
    scopes_.fill(true);
  }

  RecordFunctionCallback& needsInputs(bool needs_inputs) {
    needs_inputs_ = needs_inputs;
    return *this;
  }

  RecordFunctionCallback& samplingProb(double sampling_prob) {
    TORCH_CHECK(sampling_prob >= 0.0 && sampling_prob_ <= 1.0,
        "Invalid sampling probability");
    sampling_prob_ = sampling_prob;
    return *this;
  }

  RecordFunctionCallback& scopes(
      const std::unordered_set<RecordScope, std::hash<RecordScope>>& scopes) {
    if (!scopes.empty()) {
      scopes_.fill(false);
      for (auto sc : scopes) {
        scopes_[static_cast<size_t>(sc)] = true;
      }
    } else {
      scopes_.fill(true);
    }
    return *this;
  }

  RecordFunctionCallback& setShouldRun(
      std::function<bool(const RecordFunctionCallback&)> should_run) {
    should_run = std::move(should_run);
    return *this;
  }

  inline bool needsInputs() const {
    return needs_inputs_;
  }

  inline double samplingProb() const {
    return sampling_prob_;
  }

  inline bool checkScope(RecordScope sc) const {
    return scopes_[(size_t)sc];
  }

  inline const std::function<void(const RecordFunction&)>& start() const {
    return start_;
  }

  inline const std::function<void(const RecordFunction&)>& end() const {
    return end_;
  }

  // whether this callbacks should run in the given scope
  inline bool shouldRun(RecordScope scope) const {
    // first check whether this callback is interested in
    // the given scope type
    if (!checkScope(scope)) {
      return false;
    }
    // if we have registered should_run_ function, use it
    if (should_run_) {
      return should_run_(*this);
    }
    // otherwise potentially do the uniform sampling
    if (sampling_prob_ != 1.0) {
      return (sample_zero_one() < sampling_prob_);
    }
    return true;
  }

 private:
  std::function<void(const RecordFunction&)> start_;
  std::function<void(const RecordFunction&)> end_;
  std::function<bool(const RecordFunctionCallback&)> should_run_;
  bool needs_inputs_ = false;
  double sampling_prob_ = 1.0;
  std::array<bool, static_cast<size_t>(RecordScope::NUM_SCOPES)> scopes_ = {};

  static double sample_zero_one() {
    static thread_local auto gen =
        torch::make_unique<std::mt19937>(std::random_device()());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(*gen);
  }
};

// Using macro to minimize inputs copies,
// optional argument - function's seq_no
#define RECORD_FUNCTION_WITH_SCOPE(scope, fn, inputs, ...) \
  torch::autograd::profiler::RecordFunction guard(scope); \
  if (guard.active_) { \
    guard._setCurrent(); \
    if (guard.needs_inputs_) { \
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

// Notes:
//  - two types of callbacks are provided: thread local and global
//     - thread local callbacks are added/removed only for the given thread
//       and are stored locally for each thread and separately from the list
//       of the global callbacks
//     - global callbacks are stored in a single per process list and are
//       invoked by every RecordFunction, in addition to the thread local
//       callbacks specific to the given thread
//  - we allow the added callbacks to be sampled, by specifying a sampling
//    probability for each callback pair, if the start callback is
//    not picked to run, the corresponding end callback won't be called
//  - a typical use case for the global callbacks is passive monitoring
//    in the background (e.g. fleet-wide monitoring), without focusing on
//    the specific peice of code
//  - in contrast, thread local callbacks are enabled locally, on demand,
//    for the specific piece of code (range) and are not sampled
//  - a typical use case for thread local callbacks is profiler and code
//    execution tracer
//     - note, some functionality (e.g. profiler) can automatically
//       propagate its calbacks across thread by using ThreadLocalState
//       mechanism, but in general callbacks are not propagated
//  - adding/removing global callbacks is not thread safe and should be done
//    only when no other code is running, e.g. during the initialization

/**
 * addThreadLocalCallback adds a thread local callback to run with RecordFunction,
 * returns handle to use with removeThreadLocalCallback
 */
TORCH_API uint64_t addThreadLocalCallback(RecordFunctionCallback cb);

/**
 * hasThreadLocalCallbacks returns whether there're callbacks registered
 * with addThreadLocalCallback
 */
TORCH_API bool hasThreadLocalCallbacks();

/**
 * clearThreadLocalCallbacks removes all thread local callbacks
 */
TORCH_API void clearThreadLocalCallbacks();

/**
 * addGlobalCallback adds a global callback to run with RecordFunction:
 *
 * WARNING: not thread safe, typically addGlobalCallback can be called
 * only during the program initialization
 */
TORCH_API uint64_t addGlobalCallback(RecordFunctionCallback cb);

/**
 * removeCallback removes a callback given the handle returned by
 * addThreadLocalCallback or addGlobalCallback;
 *
 * WARNING: removing a global callback is not thread safe,
 * no other code can run simultaneously
 */
TORCH_API void removeCallback(uint64_t handle);

/**
 * hasGlobalCallbacks returns whether there're global callbacks
 * registered with pushGlobalCallback
 */
TORCH_API bool hasGlobalCallbacks();

/**
 * clearGlobalCallbacks removes all global callbacks
 * WARNING: not thread safe
 */
TORCH_API void clearGlobalCallbacks();

// for both thread local and global callbacks
TORCH_API bool hasCallbacks();
TORCH_API void clearCallbacks(); // not thread safe

/**
 * enableRecordFunction enables RecordFunction thread locally
 */
TORCH_API void enableRecordFunction(bool enable = true);

/**
 * isRecordFunctionEnabled returns whether RecordFunction
 * is enabled thread locally
 */
TORCH_API bool isRecordFunctionEnabled();

class TORCH_API RecordFunctionGuard {
 public:
  explicit RecordFunctionGuard(bool is_enabled = true)
      : prev_value_(isRecordFunctionEnabled()) {
    enableRecordFunction(is_enabled);
  }

  virtual ~RecordFunctionGuard() {
    enableRecordFunction(prev_value_);
  }

 private:
  bool prev_value_ = false;
};

class TORCH_API DisableRecordFunctionGuard : public RecordFunctionGuard {
 public:
  DisableRecordFunctionGuard() : RecordFunctionGuard(false) {}
  virtual ~DisableRecordFunctionGuard() {}
};

} // namespace profiler
} // namespace autograd
} // namespace torch
