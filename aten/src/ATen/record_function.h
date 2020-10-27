#pragma once

#include <ATen/core/ivalue.h>
#include <c10/util/SmallVector.h>
#include <c10/macros/Export.h>
#include <memory>

#include <functional>

namespace at {

// Kind of record function scope;
enum class C10_API_ENUM RecordScope : uint8_t {
  // c10/ATen ops, autograd nodes
  FUNCTION = 0,
  // Functions/nodes called from the autograd
  BACKWARD_FUNCTION,
  // TorchScript functions, methods
  TORCHSCRIPT_FUNCTION,
  // User defined scope (e.g. with record_function())
  USER_SCOPE,
  NUM_SCOPES, // must be the last in the list
};

} // namespace at

namespace std {
template <>
struct hash<at::RecordScope> {
  inline size_t operator()(
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
constexpr std::size_t kSoftLimitCallbacks = 4;

// An abstract base class for various observer contexts that can be attached to
// the RecordFunction.
struct ObserverContext {
  virtual ~ObserverContext() {}
 protected:
  ObserverContext() {}
};

typedef c10::SmallVector<uint64_t, kSoftLimitCallbacks> CallbackHandles;
typedef std::vector<std::unique_ptr<ObserverContext>> ObserverContextList;
typedef uint64_t RecordFunctionHandle;

struct TORCH_API RecordFunction {
  // Default constructor is used with before function called afterwards:
  //  scope - record scope that this function tracks
  RecordFunction(
      RecordScope scope = RecordScope::FUNCTION);

  template <typename F>
  void before(
      F fn,
      const std::vector<c10::IValue>* args,
      int64_t current_sequence_nr = -1) {
    inputs_ = *args;
    before(fn, current_sequence_nr);
  }

  // Destructor calls end callbacks
  virtual ~RecordFunction();

  RecordFunction(const RecordFunction&) = delete;
  RecordFunction& operator=(const RecordFunction&) = delete;

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
  inline uint64_t threadId() const {
    return thread_id_;
  }

  // For backward functions - thread id of the corresponding forward function,
  // or zero otherwise;
  // used alongside with sequence number to correlate backward functions with
  // the forward ones
  inline uint64_t forwardThreadId() const {
    return fwd_thread_id_;
  }

  inline void setForwardThreadId(uint64_t thread_id) {
    fwd_thread_id_ = thread_id;
  }

  inline RecordScope scope() const {
    return scope_;
  }

  // Returns logical thread_id for the current thread
  static uint64_t currentThreadId();

  // Internal functions, do not use directly;
  // used in python's context manager

  // before functions initialize RecordFunction members and call
  // start callbacks
  void before(const char* name, int64_t sequence_nr = -1);
  void before(std::string name, int64_t sequence_nr = -1);

  // Sets node ID for distributed profiling
  static void setDefaultNodeId(int64_t defaultNodeId);
  // Gets node ID for distributed profiling
  static int64_t getDefaultNodeId();

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

  // Calls end callbacks
  void end();

  inline RecordFunctionHandle handle() const {
    return handle_;
  }

  inline void setHandle(RecordFunctionHandle handle) {
    handle_ = handle;
  }

  // Whether this RecordFunction runs any callbacks
  bool active = false;
  // Whether any of the picked callbacks require inputs
  bool needs_inputs = false;

 private:
  // Allows the modification of some internal states for callbacks.
  friend class CallbackManager;

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

  // In cases when RecordFunction might be active but we chose not to
  // use the observers (e.g. operator is not observed), this boolean
  // flag is used to check whether the start callbacks were called
  bool called_start_callbacks_ = false;

  StringView name_;
  int64_t sequence_nr_ = -1;
  std::vector<c10::IValue> inputs_;

  // Kind of scope this RecordFunction is observing
  const RecordScope scope_;

  // The logical thread_id that this RecordFunction was created with
  uint64_t thread_id_ = 0;

  // For backward functions - thread id of the the forward function
  uint64_t fwd_thread_id_ = 0;

  // Unique id for this RecordFunction, used in callbacks to track start
  // and end of ranges
  RecordFunctionHandle handle_ {0};
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
  // This interface supports observers that require passing an ObserverContext
  // between start and end callbacks.
  explicit RecordFunctionCallback(
      std::function<std::unique_ptr<ObserverContext>(const RecordFunction&)> start,
      std::function<void(const RecordFunction&, ObserverContext*)> end =
        [](const RecordFunction&, ObserverContext*) {}):
      start_(std::move(start)),
      end_(std::move(end)) {
    scopes_.fill(true);
  }

  // This interface is for observers that do not pass an ObserverContext object
  // between start and end callbacks.
  explicit RecordFunctionCallback(
      std::function<void(const RecordFunction&)> start,
      std::function<void(const RecordFunction&)> end =
        [](const RecordFunction&) {}):
      start_{[start](const RecordFunction& rf) { start(rf); return nullptr; }},
      end_{[end](const RecordFunction& rf, ObserverContext*) { end(rf); }} {
    scopes_.fill(true);
  }

  RecordFunctionCallback& needsInputs(bool needs_inputs) {
    needs_inputs_ = needs_inputs;
    return *this;
  }

  RecordFunctionCallback& needsIds(bool needs_ids) {
    needs_ids_ = needs_ids;
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
    should_run_ = std::move(should_run);
    return *this;
  }

  inline bool needsInputs() const {
    return needs_inputs_;
  }

  inline bool needsIds() const {
    return needs_ids_;
  }

  inline double samplingProb() const {
    return sampling_prob_;
  }

  inline bool checkScope(RecordScope sc) const {
    return scopes_[(size_t)sc];
  }

  inline const std::function<std::unique_ptr<ObserverContext>(const RecordFunction&)>& start() const {
    return start_;
  }

  inline const std::function<void(const RecordFunction&, ObserverContext*)>& end() const {
    return end_;
  }

  // whether the callbacks should run in the given scope
  bool shouldRun(RecordScope scope) const;

 private:
  std::function<std::unique_ptr<ObserverContext>(const RecordFunction&)> start_;
  std::function<void(const RecordFunction&, ObserverContext*)> end_;
  std::function<bool(const RecordFunctionCallback&)> should_run_;
  bool needs_inputs_ = false;
  bool needs_ids_ = false;
  double sampling_prob_ = 1.0;
  std::array<bool, static_cast<size_t>(RecordScope::NUM_SCOPES)> scopes_ = {};
};

// Using macro to minimize inputs copies,
// optional argument - function's seq_no
#define RECORD_FUNCTION_WITH_SCOPE(scope, fn, inputs, ...) \
  at::RecordFunction guard(scope); \
  if (guard.active) { \
    if (guard.needs_inputs) { \
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

// Holds pairs (callbacks, unique_id)
typedef std::vector<std::pair<RecordFunctionCallback, CallbackHandle>>
    RecordFunctionCallbacks;

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

// Internal, used in ThreadLocalState to propagate TLS callbacks across threads
TORCH_API RecordFunctionCallbacks _getTLSCallbacks();
TORCH_API void _setTLSCallbacks(const RecordFunctionCallbacks& callbacks);

struct TORCH_API RecordFunctionTLS {
  // Thread local vector of callbacks, holds pairs (callbacks, unique_id);
  // must be sorted in increasing handles order
  RecordFunctionCallbacks sorted_tls_callbacks_;

  bool tls_record_function_enabled_ = true;
};

TORCH_API const RecordFunctionTLS& get_record_function_tls_();

TORCH_API void set_record_function_tls_(const RecordFunctionTLS& tls);

} // namespace at
