#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/core/operator_name.h>
#include <c10/macros/Export.h>
#include <c10/util/Optional.h>
#include <c10/util/SmallVector.h>

#include <array>
#include <atomic>
#include <functional>
#include <memory>

namespace c10 {
class TORCH_API OperatorHandle;
}

namespace at {

// Kind of record function scope;
enum class C10_API_ENUM RecordScope : uint8_t {
  // c10/ATen ops, autograd nodes
  FUNCTION = 0,
  // Functions/nodes called from the autograd
  BACKWARD_FUNCTION,
  // TorchScript functions, methods
  TORCHSCRIPT_FUNCTION,
  // Kernel Function dtype Tag
  KERNEL_FUNCTION_DTYPE,
  // Torchbind custom class,
  CUSTOM_CLASS,
  // Generic Build Feature
  BUILD_FEATURE,
  // Kernel Function dtype Tag
  LITE_INTERPRETER,
  // User defined scope (e.g. with record_function())
  USER_SCOPE,
  // Scopes for static runtime, a specialized TorchScript interpreter
  STATIC_RUNTIME_OP,
  STATIC_RUNTIME_MODEL,
  NUM_SCOPES, // must be the last in the list
};

} // namespace at

namespace std {
template <>
struct hash<at::RecordScope> {
  size_t operator()(
      const at::RecordScope& sc) const {
    return static_cast<std::size_t>(sc);
  }
};
} // namespace std

namespace at {

struct TORCH_API StringView {
  StringView() : StringView(nullptr) {}
  explicit StringView(const char* str_ptr)
    : owned_str_ptr_(nullptr), str_ptr_(str_ptr) {}
  explicit StringView(std::string str)
    : owned_str_ptr_(std::make_shared<std::string>(std::move(str))),
      str_ptr_(owned_str_ptr_->c_str()) {}

  const char* str() const {
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
constexpr std::size_t kSoftLimitCallbacks = 4;

// An abstract base class for various observer contexts that can be attached to
// the RecordFunction.
struct ObserverContext {
  virtual ~ObserverContext() {}
 protected:
  ObserverContext() {}
};

typedef c10::SmallVector<uint64_t, kSoftLimitCallbacks> CallbackHandles;
typedef c10::SmallVector<std::unique_ptr<ObserverContext>, kSoftLimitCallbacks> ObserverContextList;
typedef uint64_t RecordFunctionHandle;

struct TORCH_API RecordFunction {
  // Default constructor is used with before function called afterwards:
  //  scope - record scope that this function tracks
  //  pre_sampled - whether this RecordFunction was already pre-sampled with
  //    kLowProb probability
  RecordFunction(
      RecordScope scope = RecordScope::FUNCTION,
      bool pre_sampled = false);

  template <typename F>
  void before(
      F fn,
      const std::vector<c10::IValue>* args,
      int64_t current_sequence_nr = -1) {
    if (!isActive()) {
      return;
    }
    state_->inputs_ = *args;
    before(fn, current_sequence_nr);
  }

  // Destructor calls end callbacks
  virtual ~RecordFunction();

  RecordFunction(const RecordFunction&) = delete;
  RecordFunction& operator=(const RecordFunction&) = delete;

  const char* name() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called name() on inactive RecordFunction");
    return state_->name_.c_str();
  }

  int64_t seqNr() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called seqNr() on inactive RecordFunction");
    return state_->sequence_nr_;
  }

  const std::vector<c10::IValue>& inputs() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called inputs() on inactive RecordFunction");
    return state_->inputs_;
  }

  const std::vector<c10::IValue>& outputs() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called outputs() on inactive RecordFunction");
    return state_->outputs_;
  }

  void setOutputs(std::vector<c10::IValue>&& outputs) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called setOutputs() on inactive RecordFunction");
    state_->outputs_ = std::move(outputs);
  }

  void setOutputs(c10::ArrayRef<c10::IValue> outputs) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called setOutputs() on inactive RecordFunction");
    state_->outputs_ = outputs.vec();
  }

  size_t num_inputs() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called num_inputs() on inactive RecordFunction");
    return state_->op_input_size;
  }

  size_t num_outputs() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called num_outputs() on inactive RecordFunction");
    return state_->op_output_size;
  }

  // Retrieves the thread_id that this RecordFunction ran start callbacks with.
  // Useful for writing thread safe end callbacks that may be potentially
  // executed in a different thread (async ops)
  uint64_t threadId() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called threadId() on inactive RecordFunction");
    return state_->thread_id_;
  }

  // For backward functions - thread id of the corresponding forward function,
  // or zero otherwise;
  // used alongside with sequence number to correlate backward functions with
  // the forward ones
  uint64_t forwardThreadId() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called forwardThreadId() on inactive RecordFunction");
    return state_->fwd_thread_id_;
  }

  void setForwardThreadId(uint64_t thread_id) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called setForwardThreadId() on inactive RecordFunction");
    state_->fwd_thread_id_ = thread_id;
  }

  RecordScope scope() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called scope() on inactive RecordFunction");
    return state_->scope_;
  }

  // Returns logical thread_id for the current thread
  static uint64_t currentThreadId();

  // Internal functions, do not use directly;
  // used in python's context manager

  // before functions initialize RecordFunction members and call
  // start callbacks
  void before(const char* name, int64_t sequence_nr = -1);
  void before(std::string name, int64_t sequence_nr = -1);
  void before(c10::OperatorHandle const& op, int64_t sequence_nr = -1);

  // Sets node ID for distributed profiling
  static void setDefaultNodeId(int64_t defaultNodeId);
  // Gets node ID for distributed profiling
  static int64_t getDefaultNodeId();

  template<typename F>
  void before(
      F fn,
      c10::ArrayRef<c10::IValue> args,
      int64_t current_sequence_nr = -1) {
    if (!isActive()) {
      return;
    }
    state_->inputs_ = args.vec();
    before(fn, current_sequence_nr);
  }

  template<typename F>
  void before(
      F fn,
      std::vector<c10::IValue>&& args,
      int64_t current_sequence_nr = -1) {
    if (!isActive()) {
      return;
    }
    state_->inputs_ = std::move(args);
    before(fn, current_sequence_nr);
  }

  // Calls end callbacks. After end(), accessors will no longer provide useful results.
  void end();

  // Internal-only, used only force async event for distributed events profiling.
  void _setAsync();

  // Returns whether this RecordFunction corresponds to an async event orn ot.
  bool isAsync() const;

  RecordFunctionHandle handle() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called handle() on inactive RecordFunction");
    return state_->handle_;
  }

  c10::optional<OperatorName> operator_name() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called operator_name() on inactive RecordFunction");
    return state_->operator_name_;
  }

  void setHandle(RecordFunctionHandle handle) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called setHandle() on inactive RecordFunction");
    state_->handle_ = handle;
  }

  // Whether this RecordFunction runs any callbacks.
  bool isActive() const {
    return state_.has_value();
  }

  bool needsInputs() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called needsInputs() on inactive RecordFunction");
    return state_->needs_inputs;
  }

  bool needsOutputs() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called needsOutputs() on inactive RecordFunction");
    return state_->needs_outputs;
  }

  int64_t debugHandle() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called debugHandle() on inactive RecordFunction");
    return state_->debug_handle_;
  }

  void setDebugHandle(int64_t debug_handle) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state_, "Called setDebugHandle() on inactive RecordFunction");
    state_->debug_handle_ = debug_handle;
  }

 private:

  // Allows the modification of some internal states for callbacks.
  friend class CallbackManager;

  struct State {
    explicit State(RecordScope scope) : scope_(scope) {}

    // Whether any of the picked callbacks require inputs
    bool needs_inputs = false;

    // Whether any of the picked callbacks require outputs
    bool needs_outputs = false;

    // In cases when RecordFunction might be active but we chose not to
    // use the observers (e.g. operator is not observed), this boolean
    // flag is used to check whether the start callbacks were called
    bool called_start_callbacks_ = false;

    // Whether the RecordFunction is pre-sampled
    bool pre_sampled_ = false;

    // Used internally to keep track of thread local and global callbacks
    // that were picked to run; must be sorted;
    CallbackHandles sorted_active_tls_handles_;
    CallbackHandles sorted_active_global_handles_;

    // Stores various ObserverContext objects with event metadata for thread local
    // callbacks.
    ObserverContextList tls_ctx_;

    // Stores various ObserverContext objects with event metadata for global
    // callbacks.
    ObserverContextList global_ctx_;

    std::string name_;
    int64_t sequence_nr_ = -1;
    std::vector<c10::IValue> inputs_;
    std::vector<c10::IValue> outputs_;

    c10::optional<c10::OperatorName> operator_name_;
    size_t op_input_size{0};
    size_t op_output_size{0};

    // Kind of scope this RecordFunction is observing
    const RecordScope scope_;

    // The logical thread_id that this RecordFunction was created with
    uint64_t thread_id_ = 0;

    // For backward functions - thread id of the the forward function
    uint64_t fwd_thread_id_ = 0;

    // Unique id for this RecordFunction, used in callbacks to track start
    // and end of ranges
    RecordFunctionHandle handle_ {0};

    // Whether this record_function corresponds to an async event or not. Async
    // events can complete in different threads or follow a future-like pattern
    // of use.
    bool is_async_{false};

    // Debug handles are used for lazy annotation of module hierarchy
    // and callstack.
    // This is specifically is useful for mobile runtime, where generated
    // debug handles can be lazily symbolicated using debug information
    int64_t debug_handle_{-1};
  };

  c10::optional<State> state_;
};

//
// PyTorch callbacks/observers API:
//

/**
 * RecordFunctionCallback represents a pair of callbacks to be used with
 * RecordFunction, members:
 *   start, end - the callbacks to run when entering and exiting the scope;
 *     optionally, the start callback may return an ObserverContext which will
 *     be passed to the end callback, use appropriate constructor accordingly.
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
  using StartCallback = std::unique_ptr<ObserverContext>(*)(const RecordFunction&);
  using EndCallback = void (*)(const RecordFunction&, ObserverContext*);

  // This interface supports observers that require passing an ObserverContext
  // between start and end callbacks.
  explicit RecordFunctionCallback(
      StartCallback start,
      EndCallback end = nullptr) :
      start_(start),
      end_(end) {
    scopes_.fill(true);
  }

  RecordFunctionCallback& needsInputs(bool needs_inputs) {
    needs_inputs_ = needs_inputs;
    return *this;
  }

  RecordFunctionCallback& needsOutputs(bool needs_outputs) {
    needs_outputs_ = needs_outputs;
    return *this;
  }

  RecordFunctionCallback& needsIds(bool needs_ids) {
    needs_ids_ = needs_ids;
    return *this;
  }

  RecordFunctionCallback& samplingProb(double sampling_prob) {
    TORCH_CHECK(sampling_prob >= 0.0 && sampling_prob <= 1.0,
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

  bool needsInputs() const {
    return needs_inputs_;
  }

  bool needsOutputs() const {
    return needs_outputs_;
  }

  bool needsIds() const {
    return needs_ids_;
  }

  double samplingProb() const {
    return sampling_prob_;
  }

  bool checkScope(RecordScope sc) const {
    return scopes_[(size_t)sc];
  }

  StartCallback start() const {
    return start_;
  }

  EndCallback end() const {
    return end_;
  }

 private:
  friend class CallbackManager;
  StartCallback start_;
  EndCallback end_;
  double sampling_prob_ = 1.0;
  std::array<bool, static_cast<size_t>(RecordScope::NUM_SCOPES)> scopes_ = {};
  bool needs_inputs_ = false;
  bool needs_outputs_ = false;
  bool needs_ids_ = false;
};

// Using macro to minimize inputs copies,
// optional argument - function's seq_no
#define RECORD_FUNCTION_WITH_SCOPE(scope, fn, inputs, ...) \
  at::RecordFunction guard(scope); \
  if (guard.isActive()) {          \
    if (guard.needsInputs()) {                 \
      guard.before(fn, inputs, ##__VA_ARGS__); \
    } else { \
      guard.before(fn, ##__VA_ARGS__); \
    } \
  }

#define RECORD_FUNCTION(fn, inputs, ...) \
  RECORD_FUNCTION_WITH_SCOPE( \
    at::RecordScope::FUNCTION, \
    fn, inputs, ##__VA_ARGS__)

#define RECORD_TORCHSCRIPT_FUNCTION(mn, inputs) \
  RECORD_FUNCTION_WITH_SCOPE( \
    at::RecordScope::TORCHSCRIPT_FUNCTION, mn, inputs)

// Custom user scopes in C++; similar to Python's 'with record_function("..."):'
#define RECORD_USER_SCOPE(fn) \
  RECORD_FUNCTION_WITH_SCOPE( \
    at::RecordScope::USER_SCOPE, fn, {})

// RECORD_USER_SCOPE with inputs
#define RECORD_USER_SCOPE_WITH_INPUTS(fn, inputs) \
  RECORD_FUNCTION_WITH_SCOPE( \
    at::RecordScope::USER_SCOPE, fn, inputs)

// Helper macro to pass in debug handle that is used to
// post process events
#define RECORD_WITH_SCOPE_DEBUG_HANDLE_AND_INPUTS(  \
    scope, fn, debug_handle, inputs, ...)           \
    at::RecordFunction guard(scope);                \
    if (guard.isActive()) {                         \
      guard.setDebugHandle(debug_handle);           \
      if (guard.needsInputs()) {                    \
        guard.before(fn, inputs, ##__VA_ARGS__);    \
      } else {                                      \
        guard.before(fn, ##__VA_ARGS__);            \
      }                                             \
    }

// Helper macros to record LITE INTERPETER scope events with debug handles
#define RECORD_EDGE_SCOPE_WITH_DEBUG_HANDLE_AND_INPUTS(             \
    fn, debug_handle, inputs)                                       \
    RECORD_WITH_SCOPE_DEBUG_HANDLE_AND_INPUTS(                      \
        at::RecordScope::LITE_INTERPRETER, fn, debug_handle, inputs)

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
//  - note, thread local callbacks are automatically propagated with
//    ThreadLocalState across JIT continuations and async tasks (at::launch)
//  - adding/removing global callbacks is not thread safe and should be done
//    only when no other code is running, e.g. during the initialization

typedef uint64_t CallbackHandle;

// It is unnecessary to use atomic operations for enabling
// thread-local function callbacks. Moreover, it prevents saving to
// ThreadLocalState because std::atomic is non-copyable.
struct ThreadLocalRecordFunctionCallbacksEntry {
  RecordFunctionCallback callback;
  bool enabled = true;
  CallbackHandle handle;

  ThreadLocalRecordFunctionCallbacksEntry(RecordFunctionCallback&& cb, CallbackHandle h)
      : callback(std::move(cb)), handle(h) {}

  bool disable() {
    auto old = enabled;
    enabled = false;
    return old != enabled;
  }

  bool enable() {
    auto old = enabled;
    enabled = true;
    return old != enabled;
  }

  bool isEnabled() const {
    return enabled;
  }
};

// Holds pairs (callbacks, unique_id)
using ThreadLocalRecordFunctionCallbacks =
  std::vector<ThreadLocalRecordFunctionCallbacksEntry>;

/**
 * addThreadLocalCallback adds a thread local callback to run with RecordFunction,
 * returns handle to use with removeThreadLocalCallback
 */
TORCH_API CallbackHandle addThreadLocalCallback(
    RecordFunctionCallback cb);

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
TORCH_API CallbackHandle addGlobalCallback(
    RecordFunctionCallback cb);

/**
 * removeCallback removes a callback given the handle returned by
 * addThreadLocalCallback or addGlobalCallback;
 *
 * WARNING: removing a global callback is not thread safe,
 * no other code can run simultaneously
 */
TORCH_API void removeCallback(CallbackHandle handle);

/**
 * Prevent the given callback from executing. If handle is invalid,
 * does nothing.
 */
TORCH_API void disableCallback(CallbackHandle handle);

/**
 * Allow the given callback, previously disabled with disableCallback, to
 * execute again. If handle is invalid, does nothing.
 */
TORCH_API void reenableCallback(CallbackHandle handle);

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

struct TORCH_API RecordFunctionTLS {
  // Thread local vector of callbacks, holds pairs (callbacks, unique_id);
  // must be sorted in increasing handles order
  ThreadLocalRecordFunctionCallbacks sorted_tls_callbacks_;

  bool tls_record_function_enabled_ = true;

  // Stores the number of coin flips before the next successful coin flip
  int tries_left_ = 0;
};

TORCH_API const RecordFunctionTLS& get_record_function_tls_();

TORCH_API void set_record_function_tls_(const RecordFunctionTLS& tls);

// Checks whether RecordFunction should be called,
// sets boolean pointed by the argument to whether pre-sampling was used
TORCH_API bool shouldRunRecordFunction(bool*);

// The following functions are used to disable/enable pre-sampling of RecordFunction
// when high-frequency/non-sampled callbacks are added/removed.
// Note: every call to bumpRecordAllFunctions() is supposed to be matched with
// the corresponding releaseRecordAllFunctions() call.
// Note: disabling pre-sampling of RecordFunction incurs an extra overhead, since
// RecordFunction will be created for each operator call.
TORCH_API void bumpRecordAllFunctions();
TORCH_API void releaseRecordAllFunctions();
TORCH_API bool checkRecordAllFunctions();

} // namespace at
