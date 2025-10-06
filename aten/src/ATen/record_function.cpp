#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/record_function.h>
#include <c10/macros/Macros.h>
#include <c10/util/ThreadLocal.h>
#include <c10/util/overloaded.h>

#include <algorithm>
#include <cstdlib>
#include <random>

namespace at {

extern const std::string kParamCommsCallName = "record_param_comms";

namespace {

// Used to generate unique callback handles
CallbackHandle next_unique_callback_handle() {
  static std::atomic<uint64_t> unique_cb_id{1};
  return CallbackHandle(unique_cb_id++);
}

RecordFunctionHandle next_unique_record_function_handle() {
  static std::atomic<uint64_t> unique_rf_id{1};
  return RecordFunctionHandle(unique_rf_id++);
}

std::atomic<int64_t> defaultNodeId(-1);

// Enumerates thread ids logically;
// note: std::this_thread::get_id may return potentially
// reused thread id
std::atomic<uint64_t> next_thread_id_{0};
thread_local uint64_t current_thread_id_ = 0;

static constexpr size_t NumRecordScopes =
    static_cast<size_t>(RecordScope::NUM_SCOPES);

RecordFunctionCallbacks::iterator findCallback(
    RecordFunctionCallbacks& entries,
    CallbackHandle handle) {
  auto match_handle = [handle](const auto& el) { return el.handle_ == handle; };
  return std::find_if(entries.begin(), entries.end(), match_handle);
}

std::optional<RecordFunctionCallback> extractCallback(
    RecordFunctionCallbacks& entries,
    CallbackHandle handle) {
  auto it = findCallback(entries, handle);
  if (it == entries.end()) {
    return std::nullopt;
  }
  auto out = it->callback_;
  entries.erase(it);
  return out;
}

// ============================================================================
// == Callback manager ========================================================
// ============================================================================
// The high level idea of the RecordFunction callback machinery is based on the
// observation that the set of callbacks to be run changes infrequently.
// However, in order to reuse the active set we have to be able to invalidate
// when the active set changes. There are three events that can change which
// callbacks should be run:
//  1) The set of global callbacks changes
//  2) The set of local callbacks changes
//  3) A sampling callback is present, and should run on this iteration
//
// Global callbacks rely on thread local replication and an atomic version
// counter to maintain consistency. Whenever we change the set of active global
// callbacks (add / remove / enable / disable) the `GlobalCallbackManager`
// increments the version number and updates the global state while holding
// a mutex. The local callback manager snapshots the global callbacks and
// lazily rebuilds by comparing`GlobalCallbackManager::version()` (which is
// a simple atomic read) to the version of the last rebuild. In the
// overwhelmingly common case that they match it can reuse the existing
// snapshot. Otherwise it must call the much more expensive (and locked)
// `GlobalCallbackManager::getSnapshot()`.
//
// Handling changes to the thread local callbacks is trivial; functions that
// change them can simply force a cache rebuild for that thread after the
// changes are made.
//
// Sampling is by far the most challenging to handle efficiently. In general
// sampling callbacks are expected to have very low frequency. (e.g. 1 per
// million) Random number generation is rather expensive, so flipping a coin on
// every call for every sampling callback is wasteful. We can significantly
// reduce this cost by noting that the number of failures of a Bernoulli random
// variable is a geometric distribution, and thus we can sample the geometric
// distribution to determine the next time a callback should run. This reduces
// the cost from a random sample to a simple integer decrement.
//
// We can further note that Bernoulli samples are independent. (In contrast to,
// say, sampling without replacement.) This means that we can generate a
// counter for each scope that a given callback supports and then decrement the
// counter corresponding to the RecordScope being called. Conceptually, this is
// analogous to flipping different coins with the same probability. By sharding
// on RecordScope, we can consolidate the decrement to a single shared counter
// and update individual counters during rebuild.

class GlobalCallbackManager {
 public:
  static GlobalCallbackManager& get(); // Singleton

 private:
  GlobalCallbackManager() = default;

 public:
  static constexpr size_t NoVersion = 0;
  using snapshot_t = std::pair<size_t, RecordFunctionCallbacks>;

  //                                                                Locking?
  size_t version() const; //                                     No
  snapshot_t getSnapshot() const; //                                Yes
  CallbackHandle addCallback(RecordFunctionCallback cb); //         Yes
  void setCallbackEnabled(CallbackHandle handle, bool enabled); //  Yes
  void removeCallback(CallbackHandle handle); //                    Yes
  void clearCallbacks(); //                                         Yes

 private:
  std::atomic<size_t> version_{NoVersion + 1};
  RecordFunctionCallbacks global_callbacks_; // Source of truth.
  mutable std::mutex update_mutex_;
};

class CacheEntry {
 public:
  CacheEntry() = default;
  CacheEntry(std::mt19937* generator, RecordScope scope);

  // The caller is expected to check `GlobalCallbackManager::get().version()'
  // and call CacheEntry::update() if necessary.
  StepCallbacks getActiveCallbacks();
  std::optional<StepCallbacks> getActiveCallbacksUnlessEmpty();

  // Full rebuild. (E.g. during registration)
  void update(const std::vector<RecordFunctionCallback>& callbacks);

 private:
  struct CallbackAndCounter {
    RecordFunctionCallback callback_;

    // `-1` indicates that a callback is not sampled.
    int tries_left_{-1};
  };

  C10_ALWAYS_INLINE void getActiveCallbacksImpl();

  void rebuildActiveCallbacks();
  int sampleTries(double p) const;

  // std::mt19937 is quite large, so all scopes share the same generator.
  std::mt19937* generator_{nullptr};

  // Includes sampling callbacks which are waiting to run.
  c10::SmallVector<CallbackAndCounter, kSoftLimitCallbacks> callbacks_;
  RecordScope scope_{RecordScope::FUNCTION};

  StepCallbacks active_callbacks_;

  // For managing sampling callbacks
  int sampling_countdown_{0};
  int steps_for_this_update_{0};
};

class LocalCallbackManager {
 public:
  static LocalCallbackManager& get(); // Singleton

 private:
  LocalCallbackManager();

 public:
  const RecordFunctionTLS& getTLS() const;
  StepCallbacks getActiveCallbacks(const RecordScope scope);
  std::optional<StepCallbacks> getActiveCallbacksUnlessEmpty(
      const RecordScope scope);

  void setTLS(const RecordFunctionTLS& tls);
  void seed(uint32_t seed);
  CallbackHandle addCallback(RecordFunctionCallback callback);
  bool setCallbackEnabled(CallbackHandle handle, bool enabled);
  bool removeCallback(CallbackHandle handle);
  void clearCallbacks();

 private:
  void rebuildActiveCallbacksIfNeeded();

  void rebuild_all(const GlobalCallbackManager::snapshot_t& global_snapshot);

  void rebuild_callback_scopes(
      const GlobalCallbackManager::snapshot_t& global_snapshot,
      const RecordFunctionCallback& callback);

  void rebuild_scope(
      const GlobalCallbackManager::snapshot_t& global_snapshot,
      const RecordScope scope);

  // Source of truth.
  RecordFunctionTLS registered_callbacks_;

  // Runtime cache.
  size_t global_version_{GlobalCallbackManager::NoVersion};
  std::array<CacheEntry, NumRecordScopes> active_callbacks_;
  std::mt19937 generator_;
};

// ============================================================================
// == GlobalCallbackManager: Implementation ===================================
// ============================================================================
GlobalCallbackManager& GlobalCallbackManager::get() {
  static GlobalCallbackManager manager;
  return manager;
}

size_t GlobalCallbackManager::version() const {
  return version_.load(std::memory_order_relaxed);
}

std::pair<size_t, RecordFunctionCallbacks> GlobalCallbackManager::getSnapshot()
    const {
  std::lock_guard<std::mutex> guard(update_mutex_);
  return {version_.load(std::memory_order_seq_cst), global_callbacks_};
}

CallbackHandle GlobalCallbackManager::addCallback(RecordFunctionCallback cb) {
  std::lock_guard<std::mutex> guard(update_mutex_);
  ++version_;
  auto handle = next_unique_callback_handle();
  global_callbacks_.emplace_back(cb, handle);
  return handle;
}

void GlobalCallbackManager::setCallbackEnabled(
    CallbackHandle handle,
    bool enabled) {
  std::lock_guard<std::mutex> guard(update_mutex_);
  auto it = findCallback(global_callbacks_, handle);
  if (it != global_callbacks_.end()) {
    if (it->enabled_ != enabled) {
      ++version_;
      it->enabled_ = enabled;
    }
  } else {
    LOG(WARNING) << "Requested callback is not found";
  }
}

void GlobalCallbackManager::removeCallback(CallbackHandle handle) {
  std::lock_guard<std::mutex> guard(update_mutex_);
  if (extractCallback(global_callbacks_, handle).has_value()) {
    ++version_;
  } else {
    LOG(WARNING) << "Requested callback is not found";
  }
}

void GlobalCallbackManager::clearCallbacks() {
  std::lock_guard<std::mutex> guard(update_mutex_);
  ++version_;
  global_callbacks_.clear();
}

// ============================================================================
// == CacheEntry: Implementation ==============================================
// ============================================================================
CacheEntry::CacheEntry(std::mt19937* generator, RecordScope scope)
    : generator_{generator}, scope_{scope} {
  rebuildActiveCallbacks();
}

void CacheEntry::update(const std::vector<RecordFunctionCallback>& callbacks) {
  callbacks_.clear();
  callbacks_.reserve(callbacks.size());
  for (const auto& callback : callbacks) {
    const auto p = callback.samplingProb();
    callbacks_.push_back({callback, p < 1.0 ? sampleTries(p) : -1});
  }

  rebuildActiveCallbacks();
}

void CacheEntry::getActiveCallbacksImpl() {
  // We rebuild the active set when `sampling_countdown_` reaches zero, so if it
  // reaches zero at the start of this function something has gone wrong.
  TORCH_INTERNAL_ASSERT(sampling_countdown_ > 0, sampling_countdown_);

  if (C10_UNLIKELY(!(--sampling_countdown_))) {
    // Use inferred steps to update sampled callbacks.
    for (auto& i : callbacks_) {
      if (i.tries_left_ > 0) {
        TORCH_INTERNAL_ASSERT(i.tries_left_ >= steps_for_this_update_);
        i.tries_left_ -= steps_for_this_update_;
      }
    }

    // Determine which callbacks to run and for how long.
    rebuildActiveCallbacks();

    // Resample any sampled callbacks that ran this call.
    for (auto& i : callbacks_) {
      if (!i.tries_left_) {
        i.tries_left_ = sampleTries(i.callback_.samplingProb());
      }
    }
  }
}

StepCallbacks CacheEntry::getActiveCallbacks() {
  getActiveCallbacksImpl();
  return active_callbacks_;
}

std::optional<StepCallbacks> CacheEntry::getActiveCallbacksUnlessEmpty() {
  getActiveCallbacksImpl();
  if (C10_LIKELY(active_callbacks_.empty())) {
    return std::nullopt;
  }
  return active_callbacks_;
}

void CacheEntry::rebuildActiveCallbacks() {
  // We could store thread ID in CacheEntry, but rebuilds are infrequent and
  // this saves us from having to plumb it through.
  const auto thread_id = RecordFunction::currentThreadId();
  active_callbacks_ = StepCallbacks(thread_id, scope_);

  sampling_countdown_ = std::numeric_limits<int>::max();
  for (const auto& i : callbacks_) {
    if (i.tries_left_ < 0) {
      // Callback is not sampled. Unconditionally push.
      active_callbacks_.callbacks_.push_back(
          {i.callback_.start(), i.callback_.end()});

    } else if (i.tries_left_ == 0) {
      // Callback is sampled and we have reached a sampling event. Push and
      // set `sampling_countdown_` to one so we trigger a rebuild after one
      // call.
      active_callbacks_.callbacks_.push_back(
          {i.callback_.start(), i.callback_.end()});
      sampling_countdown_ = 1;

    } else {
      // Callback is sampled and we have not reached sampling event. Set
      // `sampling_countdown_` to rebuild when it is time for this callback to
      // execute.
      sampling_countdown_ = std::min(sampling_countdown_, i.tries_left_);
    }
    active_callbacks_.needs_inputs_ |= i.callback_.needsInputs();
    active_callbacks_.needs_outputs_ |= i.callback_.needsOutputs();
    active_callbacks_.needs_ids_ |= i.callback_.needsIds();
  }
  steps_for_this_update_ = sampling_countdown_;
}

int CacheEntry::sampleTries(double p) const {
  TORCH_INTERNAL_ASSERT(generator_ != nullptr);
  TORCH_INTERNAL_ASSERT(p > 0.0 && p <= 1.0);

  // The geometric distribution returns the number of failures. We add one to
  // also account for the call where we succeed.
  return std::geometric_distribution<int>(p)(*generator_) + 1;
}

// ============================================================================
// == LocalCallbackManager: Implementation ====================================
// ============================================================================
LocalCallbackManager& LocalCallbackManager::get() {
#if defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
  static c10::ThreadLocal<LocalCallbackManager> manager;
  return manager.get();
#else // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
  static thread_local LocalCallbackManager manager;
  return manager;
#endif // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
}

LocalCallbackManager::LocalCallbackManager() {
  for (auto i : c10::irange(NumRecordScopes)) {
    active_callbacks_[i] = CacheEntry(&generator_, static_cast<RecordScope>(i));
  }
  rebuild_all(GlobalCallbackManager::get().getSnapshot());
}

const RecordFunctionTLS& LocalCallbackManager::getTLS() const {
  return registered_callbacks_;
}

void LocalCallbackManager::rebuildActiveCallbacksIfNeeded() {
  const auto global_version = GlobalCallbackManager::get().version();
  if (C10_UNLIKELY(global_version != global_version_)) {
    rebuild_all(GlobalCallbackManager::get().getSnapshot());
  }
}

StepCallbacks LocalCallbackManager::getActiveCallbacks(
    const RecordScope scope) {
  rebuildActiveCallbacksIfNeeded();
  return active_callbacks_[static_cast<size_t>(scope)].getActiveCallbacks();
}

std::optional<StepCallbacks> LocalCallbackManager::
    getActiveCallbacksUnlessEmpty(const RecordScope scope) {
  rebuildActiveCallbacksIfNeeded();
  return active_callbacks_[static_cast<size_t>(scope)]
      .getActiveCallbacksUnlessEmpty();
}

void LocalCallbackManager::setTLS(const RecordFunctionTLS& tls) {
  registered_callbacks_ = tls;
  rebuild_all(GlobalCallbackManager::get().getSnapshot());
}

void LocalCallbackManager::seed(uint32_t seed) {
  generator_.seed(seed);
}

CallbackHandle LocalCallbackManager::addCallback(
    RecordFunctionCallback callback) {
  auto handle = next_unique_callback_handle();
  auto& callbacks = registered_callbacks_.sorted_tls_callbacks_;
  callbacks.emplace_back(callback, handle);
  rebuild_callback_scopes(
      GlobalCallbackManager::get().getSnapshot(), callbacks.back().callback_);
  return handle;
}

bool LocalCallbackManager::setCallbackEnabled(
    CallbackHandle handle,
    bool enabled) {
  auto it = findCallback(registered_callbacks_.sorted_tls_callbacks_, handle);
  auto found = (it != registered_callbacks_.sorted_tls_callbacks_.end());
  if (found && it->enabled_ != enabled) {
    it->enabled_ = enabled;
    rebuild_callback_scopes(
        GlobalCallbackManager::get().getSnapshot(), it->callback_);
  }
  return found;
}

bool LocalCallbackManager::removeCallback(CallbackHandle handle) {
  auto& callbacks = registered_callbacks_.sorted_tls_callbacks_;
  auto callback = extractCallback(callbacks, handle);
  if (callback.has_value()) {
    rebuild_callback_scopes(
        GlobalCallbackManager::get().getSnapshot(), *callback);
  }
  return callback.has_value();
}

void LocalCallbackManager::clearCallbacks() {
  registered_callbacks_.sorted_tls_callbacks_.clear();
  rebuild_all(GlobalCallbackManager::get().getSnapshot());
}

void LocalCallbackManager::rebuild_all(
    const GlobalCallbackManager::snapshot_t& global_snapshot) {
  global_version_ = global_snapshot.first;
  for (auto i : c10::irange(NumRecordScopes)) {
    rebuild_scope(global_snapshot, static_cast<RecordScope>(i));
  }
}

void LocalCallbackManager::rebuild_callback_scopes(
    const GlobalCallbackManager::snapshot_t& global_snapshot,
    const RecordFunctionCallback& callback) {
  if (global_snapshot.first == global_version_) {
    // Only rebuild scopes associated with `callback`
    for (auto i : c10::irange(NumRecordScopes)) {
      if (callback.checkScope(static_cast<RecordScope>(i))) {
        rebuild_scope(global_snapshot, static_cast<RecordScope>(i));
      }
    }
  } else {
    rebuild_all(global_snapshot);
  }
}

void LocalCallbackManager::rebuild_scope(
    const GlobalCallbackManager::snapshot_t& global_snapshot,
    const RecordScope scope) {
  std::vector<RecordFunctionCallback> callbacks;
  if (registered_callbacks_.tls_record_function_enabled_) {
    auto populate_callbacks =
        [&](const RecordFunctionCallbacks& raw_callbacks) {
          for (const auto& i : raw_callbacks) {
            if (i.enabled_ && i.callback_.checkScope(scope) &&
                i.callback_.samplingProb() > 0) {
              callbacks.push_back(i.callback_);
            }
          }
        };
    populate_callbacks(global_snapshot.second);
    populate_callbacks(registered_callbacks_.sorted_tls_callbacks_);
  }
  active_callbacks_[static_cast<size_t>(scope)].update(callbacks);
}

// ============================================================================
// == Callback execution ======================================================
// ============================================================================
void logTryRunCallbackError(const char* what, const char* name) {
  LOG(WARNING) << "Exception in RecordFunction callback: " << what
               << " , for the range " << name;
}

template <bool is_start>
C10_ALWAYS_INLINE bool tryRunCallback(
    const StepCallbacks::StartEndPair callback_ptrs,
    const RecordFunction& rf,
    std::unique_ptr<ObserverContext>& ctx) {
  try {
    if (is_start && callback_ptrs.start_) {
      ctx = callback_ptrs.start_(rf);
    }

    if (!is_start && callback_ptrs.end_) {
      callback_ptrs.end_(rf, ctx.get());
    }

    return true;
  } catch (const std::exception& e) {
    logTryRunCallbackError(e.what(), rf.name());
    return false;
  } catch (...) {
    logTryRunCallbackError("unknown", rf.name());
    return false;
  }
}

} // namespace

RecordFunction::RecordFunction(RecordScope scope)
    : RecordFunction(getStepCallbacks(scope)) {}

RecordFunction::RecordFunction(StepCallbacks&& step_callbacks)
    : step_callbacks_{std::move(step_callbacks)} {
  ctx_.resize(step_callbacks_.callbacks_.size());
  if (step_callbacks_.needs_ids_) {
    setHandle(next_unique_record_function_handle());
  }
}

void RecordFunction::runStartCallbacks() {
  for (const auto i : c10::irange(step_callbacks_.callbacks_.size())) {
    tryRunCallback</*is_start=*/true>(
        step_callbacks_.callbacks_[i], *this, ctx_[i]);
  }
  called_start_callbacks_ = true;
}

void RecordFunction::end() {
  if (called_start_callbacks_) {
    for (const auto i : c10::irange(step_callbacks_.callbacks_.size())) {
      tryRunCallback</*is_start=*/false>(
          step_callbacks_.callbacks_[i], *this, ctx_[i]);
    }
    step_callbacks_.callbacks_.clear();
  }
}

const char* RecordFunction::name() const {
  return std::visit(
      c10::overloaded(
          [](const std::string& name) { return name.c_str(); },
          [](const schema_ref_t schema) {
            return schema.get().name().c_str();
          }),
      fn_);
}

size_t RecordFunction::num_inputs() const {
  return std::visit(
      c10::overloaded(
          [&](const std::string&) { return inputs_.size(); },
          [](const schema_ref_t schema) {
            return schema.get().arguments().size();
          }),
      fn_);
}

size_t RecordFunction::num_outputs() const {
  return std::visit(
      c10::overloaded(
          [&](const std::string&) { return outputs_.size(); },
          [](const schema_ref_t schema) {
            return schema.get().returns().size();
          }),
      fn_);
}

std::optional<OperatorName> RecordFunction::operator_name() const {
  return std::visit(
      c10::overloaded(
          [&](const std::string&) -> std::optional<OperatorName> {
            return std::nullopt;
          },
          [](const schema_ref_t schema) -> std::optional<OperatorName> {
            return schema.get().operator_name();
          }),
      fn_);
}

std::optional<c10::FunctionSchema> RecordFunction::operator_schema() const {
  return std::visit(
      c10::overloaded(
          [&](const std::string&) -> std::optional<c10::FunctionSchema> {
            return std::nullopt;
          },
          [](const schema_ref_t schema) -> std::optional<c10::FunctionSchema> {
            return schema.get();
          }),
      fn_);
}

const char* RecordFunction::overload_name() const {
  return std::visit(
      c10::overloaded(
          [&](const std::string&) -> const char* { return ""; },
          [](const schema_ref_t schema) -> const char* {
            return schema.get().overload_name().c_str();
          }),
      fn_);
}

StepCallbacks getStepCallbacks(RecordScope scope) {
  return LocalCallbackManager::get().getActiveCallbacks(scope);
}

std::optional<StepCallbacks> getStepCallbacksUnlessEmpty(RecordScope scope) {
  return LocalCallbackManager::get().getActiveCallbacksUnlessEmpty(scope);
}

const RecordFunctionTLS& get_record_function_tls_() {
  return LocalCallbackManager::get().getTLS();
}

void set_record_function_tls_(const RecordFunctionTLS& tls) {
  LocalCallbackManager::get().setTLS(tls);
}

namespace {
bool anyEnabled(const RecordFunctionCallbacks& callbacks) {
  return std::any_of(callbacks.begin(), callbacks.end(), [](const auto& cb) {
    return cb.enabled_;
  });
}
} // namespace

bool hasCallbacks() {
  return hasThreadLocalCallbacks() || hasGlobalCallbacks();
}

bool hasGlobalCallbacks() {
  return anyEnabled(GlobalCallbackManager::get().getSnapshot().second);
}

bool hasThreadLocalCallbacks() {
  return anyEnabled(get_record_function_tls_().sorted_tls_callbacks_);
}

CallbackHandle addThreadLocalCallback(RecordFunctionCallback cb) {
  return LocalCallbackManager::get().addCallback(cb);
}

CallbackHandle addGlobalCallback(RecordFunctionCallback cb) {
  return GlobalCallbackManager::get().addCallback(cb);
}

void removeCallback(CallbackHandle handle) {
  if (!LocalCallbackManager::get().removeCallback(handle)) {
    GlobalCallbackManager::get().removeCallback(handle);
  }
}

void disableCallback(CallbackHandle handle) {
  if (!LocalCallbackManager::get().setCallbackEnabled(handle, false)) {
    GlobalCallbackManager::get().setCallbackEnabled(handle, false);
  }
}

void reenableCallback(CallbackHandle handle) {
  if (!LocalCallbackManager::get().setCallbackEnabled(handle, true)) {
    GlobalCallbackManager::get().setCallbackEnabled(handle, true);
  }
}

void clearGlobalCallbacks() {
  GlobalCallbackManager::get().clearCallbacks();
}

void clearThreadLocalCallbacks() {
  LocalCallbackManager::get().clearCallbacks();
}

void clearCallbacks() {
  clearGlobalCallbacks();
  clearThreadLocalCallbacks();
}

bool isRecordFunctionEnabled() {
  return LocalCallbackManager::get().getTLS().tls_record_function_enabled_;
}

void enableRecordFunction(bool enable) {
  auto tls = LocalCallbackManager::get().getTLS();
  if (tls.tls_record_function_enabled_ != enable) {
    tls.tls_record_function_enabled_ = enable;
    LocalCallbackManager::get().setTLS(tls);
  }
}

void set_record_function_seed_for_testing(uint32_t seed) {
  LocalCallbackManager::get().seed(seed);
}

/* static */
uint64_t RecordFunction::currentThreadId() {
  if (!current_thread_id_) {
    // happens only once per thread
    current_thread_id_ = ++next_thread_id_;
  }
  return current_thread_id_;
}

void RecordFunction::before(RecordFunction::FunctionDescriptor fn, int64_t sequence_nr) {
  std::visit([this](auto&& fn) {
    if constexpr (std::is_same_v<std::decay_t<decltype(fn)>, std::string_view>) {
      is_nccl_meta_ = (fn == kParamCommsCallName);
      fn_ = std::string(fn);
    } else {
      is_nccl_meta_ = (fn.get().name() == kParamCommsCallName);
      fn_ = fn;
    }
  }, fn);
  sequence_nr_ = sequence_nr;

#ifndef NDEBUG
  inputs_valid_ = true;
#endif
  runStartCallbacks();
  invalidateInputs();
}

/* static */ void RecordFunction::setDefaultNodeId(int64_t newDefaultNodeId) {
  TORCH_CHECK(newDefaultNodeId >= 0, "setDefaultNodeId expects an id >= 0.");
  defaultNodeId = newDefaultNodeId;
}

/* static */ int64_t RecordFunction::getDefaultNodeId() {
  return defaultNodeId;
}

RecordFunction::~RecordFunction() {
  end();
}

void RecordFunction::_setAsync() {
  is_async_ = true;
}

bool RecordFunction::isAsync() const {
  return is_async_;
}

void RecordFunction::_setStaticRuntimeOutVariant() {
  if (isActive()) {
    is_static_runtime_out_variant_ = true;
  }
}

bool RecordFunction::isStaticRuntimeOutVariant() const {
  if (isActive()) {
    return is_static_runtime_out_variant_;
  }
  return false;
}
} // namespace at
