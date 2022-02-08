#include <ATen/record_function.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/macros/Macros.h>
#include <c10/util/ThreadLocal.h>

#include <algorithm>
#include <cstdlib>
#include <random>

namespace at {

namespace {

// Used to generate unique callback handles
CallbackHandle next_unique_callback_handle() {
  static std::atomic<uint64_t> unique_cb_id {1};
  return CallbackHandle(unique_cb_id++);
}

RecordFunctionHandle next_unique_record_function_handle() {
  static std::atomic<uint64_t> unique_rf_id {1};
  return RecordFunctionHandle(unique_rf_id++);
}

RecordFunctionTLS& rf_tls() {
#if defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
  static c10::ThreadLocal<RecordFunctionTLS> rf_tls_;
  return rf_tls_.get();
#else // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
  static thread_local RecordFunctionTLS rf_tls_;
  return rf_tls_;
#endif // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
}

std::atomic<int64_t> defaultNodeId(-1);

// Enumerates thread ids logically;
// note: std::this_thread::get_id may return potentially
// reused thread id
std::atomic<uint64_t> next_thread_id_ {0};
thread_local uint64_t current_thread_id_ = 0;

// Low probability constant
static constexpr double kLowProb = 0.001;
struct CoinflipTLS {
  int tries_left_;
  std::mt19937 genGeo_;
  std::mt19937 genZeroOne_;
  std::geometric_distribution<int> distGeo_;
  std::uniform_real_distribution<double> distZeroOne_;
  CoinflipTLS();
};

CoinflipTLS::CoinflipTLS()
    : tries_left_(0), genGeo_(std::random_device()()), genZeroOne_(std::random_device()()), distGeo_(kLowProb), distZeroOne_(0.0, 1.0) {}

CoinflipTLS& coinflip_tls() {
#if defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
  static c10::ThreadLocal<CoinflipTLS> coinflip_tls_;
  return coinflip_tls_.get();
#else // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
  static thread_local CoinflipTLS coinflip_tls_;
  return coinflip_tls_;
#endif // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
}

int sample_geometric() {
  return coinflip_tls().distGeo_(coinflip_tls().genGeo_);
}

double sample_zero_one() {
  return coinflip_tls().distZeroOne_(coinflip_tls().genZeroOne_);
}

struct GlobalRecordFunctionCallbacksEntry {
  RecordFunctionCallback callback;
 private:
  std::atomic<bool> enabled;
 public:
  CallbackHandle handle;

  GlobalRecordFunctionCallbacksEntry(RecordFunctionCallback&& cb, CallbackHandle h)
      : callback(std::move(cb)), enabled(true), handle(h) {}

  // Copying is fine despite std::atomic<bool> not being supposed to
  // have a copy/move constructor: adding & removing callbacks is
  // already not thread-safe.
  GlobalRecordFunctionCallbacksEntry(
      const GlobalRecordFunctionCallbacksEntry& rhs)
      : callback(rhs.callback), enabled(rhs.enabled.load()), handle(rhs.handle) {}

  GlobalRecordFunctionCallbacksEntry& operator=(const GlobalRecordFunctionCallbacksEntry& rhs) {
    callback = rhs.callback;
    enabled = rhs.enabled.load();
    handle = rhs.handle;
    return *this;
  }

  GlobalRecordFunctionCallbacksEntry(
      GlobalRecordFunctionCallbacksEntry&& rhs) noexcept
      : callback(std::move(rhs.callback)), enabled(rhs.enabled.load()), handle(rhs.handle) {}

  GlobalRecordFunctionCallbacksEntry& operator=(GlobalRecordFunctionCallbacksEntry&& rhs) noexcept {
    callback = std::move(rhs.callback);
    enabled = rhs.enabled.load();
    handle = rhs.handle;
    return *this;
  }

  // Returns true if the status changed, false otherwise.
  bool disable() {
    bool expected = true;
    // NOTE: we use sequentially consistent access here and in
    // enable() because updating further atomic flags depends on this
    // operation.
    return enabled.compare_exchange_strong(expected, false);
  }

  // Returns true if the status changed, false otherwise.
  bool enable() {
    bool expected = false;
    return enabled.compare_exchange_strong(expected, true);
  }

  // Read the flag. Note that it is neither necessary nor correct to
  // check this before calling enable() or disable().
  bool isEnabled() const {
    return enabled.load(std::memory_order_relaxed);
  }
};

using GlobalRecordFunctionCallbacks =
  c10::SmallVector<GlobalRecordFunctionCallbacksEntry, kSoftLimitCallbacks>;

} // namespace

const RecordFunctionTLS& get_record_function_tls_() {
  return rf_tls();
}

void set_record_function_tls_(const RecordFunctionTLS& tls) {
  rf_tls() = tls;
}

enum class ToggledCallbackResult {
  NotFound,
  FoundButNotToggled,
  FoundAndToggled,
};

template <typename RecordFunctionCallbacks>
static ToggledCallbackResult findAndToggleCallback(
    RecordFunctionCallbacks& cbs, CallbackHandle handle, bool enabled) {
  auto it = std::find_if(
      cbs.begin(), cbs.end(),
      [handle](
          const auto& el) {
        return el.handle == handle;
      });
  if (it != cbs.end()) {
    bool changed = enabled ? it->enable() : it->disable();
    if (!changed) {
      return ToggledCallbackResult::FoundButNotToggled;
    }
    if (it->callback.samplingProb() > kLowProb) {
      // try to disable/restore pre-sampling of RecordFunction
      if (enabled) {
        at::bumpRecordAllFunctions();
      } else {
        at::releaseRecordAllFunctions();
      }
    }
    return ToggledCallbackResult::FoundAndToggled;
  }
  return ToggledCallbackResult::NotFound;
}

template <typename RecordFunctionCallbacks>
static bool findAndRemoveCallback(
    RecordFunctionCallbacks& cbs, CallbackHandle handle) {
  auto it = std::find_if(
      cbs.begin(), cbs.end(),
      [handle](
          const auto& el) {
        return el.handle == handle;
      });
  if (it != cbs.end()) {
    // We do not need to try to call releaseRecordAllFunctions here
    // because findAndRemoveCallback is used only as a helper in
    // removeCallback. removeCallback calls disableCallback, which
    // calls findAndToggleCallback, which already will do a
    // releaseRecordAllFunctions for us.
    cbs.erase(it);
    return true;
  }
  return false;
}

class CallbackManager {
 public:
  CallbackManager() : num_enabled_global_callbacks_(0) {}

  CallbackHandle addThreadLocalCallback(RecordFunctionCallback cb) {
    if (cb.samplingProb() > kLowProb) {
      // pre-sampling of RecordFunction with prob. kLowProb cannot be used
      at::bumpRecordAllFunctions();
    }
    // note: monotonically increasing callbacks_unique_id keeps
    // sorted_tls_callbacks_ sorted
    auto handle = next_unique_callback_handle();
    rf_tls().sorted_tls_callbacks_.emplace_back(std::move(cb), handle);
    return handle;
  }

  CallbackHandle addGlobalCallback(RecordFunctionCallback cb) {
    if (cb.samplingProb() > kLowProb) {
      // pre-sampling of RecordFunction with prob. kLowProb cannot be used
      at::bumpRecordAllFunctions();
    }
    auto handle = next_unique_callback_handle();
    // NOLINTNEXTLINE(performance-move-const-arg)
    sorted_global_callbacks_.emplace_back(std::move(cb), handle);
    num_enabled_global_callbacks_.fetch_add(1, std::memory_order_relaxed);
    return handle;
  }

  void removeCallback(CallbackHandle handle) {
    // This could be implemented more efficiently, but callback
    // addition/removal is not intended to run in performance-critical
    // paths (it's not thread-safe and should be done during
    // initialization).
    disableCallback(handle);
    auto found = findAndRemoveCallback(rf_tls().sorted_tls_callbacks_, handle);
    if (!found) {
      found = findAndRemoveCallback(sorted_global_callbacks_, handle);
    }
    if (!found) {
      LOG(WARNING) << "Requested callback is not found";
    }
  }

  void disableCallback(CallbackHandle handle) {
    auto found = findAndToggleCallback(
        rf_tls().sorted_tls_callbacks_, handle, false);
    if (found == ToggledCallbackResult::NotFound) {
      found = findAndToggleCallback(
          sorted_global_callbacks_, handle, false);
      if (found == ToggledCallbackResult::FoundAndToggled) {
        const auto previousCount = num_enabled_global_callbacks_.fetch_sub(1, std::memory_order_relaxed);
        TORCH_CHECK(previousCount > 0, previousCount);
      }
    }
    if (found == ToggledCallbackResult::NotFound) {
      LOG(WARNING) << "Requested callback is not found";
    }
  }

  void reenableCallback(CallbackHandle handle) {
    auto found = findAndToggleCallback(
        rf_tls().sorted_tls_callbacks_, handle, true);
    if (found == ToggledCallbackResult::NotFound) {
      found = findAndToggleCallback(
          sorted_global_callbacks_, handle, true);
      if (found == ToggledCallbackResult::FoundAndToggled) {
        num_enabled_global_callbacks_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    if (found == ToggledCallbackResult::NotFound) {
      LOG(WARNING) << "Requested callback is not found";
    }
  }

  void clearGlobalCallbacks() {
    sorted_global_callbacks_.clear();
    num_enabled_global_callbacks_ = 0;
  }

  void clearThreadLocalCallbacks() {
    rf_tls().sorted_tls_callbacks_.clear();
  }

  inline bool hasGlobalCallbacks() const {
    return num_enabled_global_callbacks_.load(std::memory_order_relaxed) > 0;
  }

  inline bool hasThreadLocalCallbacks() const {
    return !rf_tls().sorted_tls_callbacks_.empty();
  }

  // We need this function to be inlined: init() is a hot path and
  // callbackShouldRun is even hotter because it's called multiple
  // times per init(). Profiling shows that the function prologue is
  // taking up a significant fraction of the time.
  static bool C10_ALWAYS_INLINE callbackShouldRun(
      const RecordFunctionCallback& cb, RecordScope scope, bool pre_sampled) {
    TORCH_INTERNAL_ASSERT(
        !pre_sampled || (cb.sampling_prob_ <= kLowProb),
        "Incorrect usage of a pre-sampled RecordFunction with a high-frequency "
        " or non-sampled callback");

    // first check whether this callback is interested in
    // the given scope type
    if (!cb.checkScope(scope)) {
      return false;
    }

    // otherwise potentially do the sampling
    double sampling_prob = cb.sampling_prob_;
    constexpr double kLowProbInv = 1 / kLowProb;
    if (pre_sampled) {
      // adjust the sampling rate to account for kLowProb pre-sampling of
      // the RecordFunction
      sampling_prob *= kLowProbInv;
    }

    if (sampling_prob < 1.0) {
      // model the low probability events as events happening
      // with probability kLowProb followed by another sampling with
      // probability (sampling_prob / kLowProb), then replace the coin
      // flip for kLowProb with a thread local number of tries tries_left_
      // sampled from the geometric distribution.
      if (sampling_prob < kLowProb) {
        if (coinflip_tls().tries_left_ == 0) {
          coinflip_tls().tries_left_ = sample_geometric();
          return (sample_zero_one() < sampling_prob * kLowProbInv);
        } else {
          --coinflip_tls().tries_left_;
          return false;
        }
      } else {
        return (sample_zero_one() < sampling_prob);
      }
    }

    return true;
  }

  // init is called by RecordFunction in constructor to
  // determine which thread local and global callbacks are going
  // to be executed and whether any of them need inputs
  inline void init(RecordFunction& rec_fn, RecordScope scope, bool pre_sampled) {
    bool found_needs_inputs = false;
    bool found_needs_outputs = false;
    bool found_needs_ids = false;

    for (const auto& cb: rf_tls().sorted_tls_callbacks_) {
      if (cb.isEnabled() && callbackShouldRun(cb.callback, scope, pre_sampled)) {
        if (cb.callback.needsInputs()) {
          found_needs_inputs = true;
        }
        if (cb.callback.needsOutputs()) {
          found_needs_outputs = true;
        }
        if (cb.callback.needsIds()) {
          found_needs_ids = true;
        }
        if (!rec_fn.state_) {
          rec_fn.state_.emplace(scope);
        }
        rec_fn.state_->sorted_active_tls_handles_.push_back(cb.handle);
      }
    }

    for (const auto& cb: sorted_global_callbacks_) {
      if (cb.isEnabled() && callbackShouldRun(cb.callback, scope, pre_sampled)) {
        if (cb.callback.needsInputs()) {
          found_needs_inputs = true;
        }
        if (cb.callback.needsOutputs()) {
          found_needs_outputs = true;
        }
        if (cb.callback.needsIds()) {
          found_needs_ids = true;
        }
        if (!rec_fn.state_) {
          rec_fn.state_.emplace(scope);
        }
        rec_fn.state_->sorted_active_global_handles_.push_back(cb.handle);
      }
    }

    if (!rec_fn.state_) {
      return;
    }

    // Pre-allocate observer context list with nullptr.
    rec_fn.state_->tls_ctx_.resize(rec_fn.state_->sorted_active_tls_handles_.size());
    rec_fn.state_->global_ctx_.resize(rec_fn.state_->sorted_active_global_handles_.size());

    rec_fn.state_->needs_inputs = found_needs_inputs;
    rec_fn.state_->needs_outputs = found_needs_outputs;
    if (found_needs_ids) {
      rec_fn.setHandle(next_unique_record_function_handle());
    }
  }

  void runStartCallbacks(RecordFunction& rf) {
    mergeRunCallbacks(
        sorted_global_callbacks_,
        rf.state_->sorted_active_global_handles_,
        rf.state_->global_ctx_,
        /* is_start */ true,
        rf);
    mergeRunCallbacks(
        rf_tls().sorted_tls_callbacks_,
        rf.state_->sorted_active_tls_handles_,
        rf.state_->tls_ctx_,
        /* is_start */ true,
        rf);
    rf.state_->called_start_callbacks_ = true;
  }

  void runEndCallbacks(RecordFunction& rf) {
    mergeRunCallbacks(
        sorted_global_callbacks_,
        rf.state_->sorted_active_global_handles_,
        rf.state_->global_ctx_,
        /* is_start */ false,
        rf);
    mergeRunCallbacks(
        rf_tls().sorted_tls_callbacks_,
        rf.state_->sorted_active_tls_handles_,
        rf.state_->tls_ctx_,
        /* is_start */ false,
        rf);
  }

  // Global callbacks; must be sorted in increasing handle order
  GlobalRecordFunctionCallbacks sorted_global_callbacks_;
  std::atomic<uint_fast32_t> num_enabled_global_callbacks_;

 private:
  static void logTryRunCallbackError(const char* what, const RecordFunction& rf) {
    LOG(WARNING) << "Exception in RecordFunction callback: " << what << " , for the range " << rf.name();
  }

  C10_ALWAYS_INLINE static bool tryRunCallback(
      const RecordFunctionCallback& rfcb,
      RecordFunction& rf,
      std::unique_ptr<ObserverContext>& ctx,
      bool is_start) {
    try {
      if (is_start) {
        ctx = rfcb.start() ? rfcb.start()(rf) : nullptr;
      }
      else {
        if (rfcb.end()) {
          rfcb.end()(rf, ctx.get());
        }
      }
      return true;
    } catch (const std::exception &e) {
      logTryRunCallbackError(e.what(), rf);
      return false;
    } catch (...) {
      logTryRunCallbackError("unknown", rf);
      return false;
    }
  }

  template <typename RecordFunctionCallbacks>
  static void mergeRunCallbacks(
      const RecordFunctionCallbacks& sorted_callbacks,
      const CallbackHandles& sorted_handles,
      ObserverContextList& ctx_list,
      bool is_start,
      RecordFunction& rf) {
    size_t num_executed = 0;
    size_t idx_c = 0;
    const auto sorted_handles_size = sorted_handles.size();
    const auto ctx_list_size = ctx_list.size();
    const auto sorted_callbacks_size = sorted_callbacks.size();
    for (size_t idx_h = 0; idx_h < sorted_handles_size && idx_h < ctx_list_size; ++idx_h) {
      while (idx_c < sorted_callbacks_size &&
            sorted_callbacks[idx_c].handle < sorted_handles[idx_h]) {
        ++idx_c;
      }
      if (idx_c >= sorted_callbacks_size) {
        break;
      }
      if (sorted_callbacks[idx_c].handle == sorted_handles[idx_h]) {
        tryRunCallback(sorted_callbacks[idx_c].callback, rf, ctx_list[idx_h], is_start);
        ++num_executed;
      }
    }

    if (num_executed != sorted_handles.size()) {
      C10_LOG_EVERY_MS(WARNING, 1000)
          << "Could not match some of the start callbacks with the corresponding end callbacks, "
          << "callbacks changed during RecordFunction lifetime; you might be trying to profile "
          << "the code after profiler is finished";
    }
  }
};

namespace {
  // Keeping this static manager local.
  CallbackManager& manager() {
    static CallbackManager _manager;
    return _manager;
  }
} // namespace

bool hasCallbacks() {
  auto& m = manager();
  return m.hasGlobalCallbacks() || m.hasThreadLocalCallbacks();
}

bool hasGlobalCallbacks() {
  return manager().hasGlobalCallbacks();
}

bool hasThreadLocalCallbacks() {
  return manager().hasThreadLocalCallbacks();
}

CallbackHandle addThreadLocalCallback(
    RecordFunctionCallback cb) {
  // NOLINTNEXTLINE(performance-move-const-arg)
  return manager().addThreadLocalCallback(std::move(cb));
}

CallbackHandle addGlobalCallback(
    RecordFunctionCallback cb) {
  // NOLINTNEXTLINE(performance-move-const-arg)
  return manager().addGlobalCallback(std::move(cb));
}

void removeCallback(CallbackHandle handle) {
  manager().removeCallback(handle);
}

void disableCallback(CallbackHandle handle) {
  manager().disableCallback(handle);
}

void reenableCallback(CallbackHandle handle) {
  manager().reenableCallback(handle);
}

void clearGlobalCallbacks() {
  manager().clearGlobalCallbacks();
}

void clearThreadLocalCallbacks() {
  manager().clearThreadLocalCallbacks();
}

void clearCallbacks() {
  auto& m = manager();
  m.clearGlobalCallbacks();
  m.clearThreadLocalCallbacks();
}

bool isRecordFunctionEnabled() {
  return rf_tls().tls_record_function_enabled_;
}

void enableRecordFunction(bool enable) {
  rf_tls().tls_record_function_enabled_ = enable;
}

RecordFunction::RecordFunction(RecordScope scope, bool pre_sampled) {
  auto* rf_tls_ptr = &rf_tls();
  if (rf_tls_ptr->tls_record_function_enabled_) {
    auto& m = manager();
    if (!m.sorted_global_callbacks_.empty() || !rf_tls_ptr->sorted_tls_callbacks_.empty()) {
      m.init(*this, scope, pre_sampled);
    }
  }
}

/* static */
uint64_t RecordFunction::currentThreadId() {
  if (!current_thread_id_) {
    // happens only once per thread
    current_thread_id_ = ++next_thread_id_;
  }
  return current_thread_id_;
}

void RecordFunction::before(const char* name, int64_t sequence_nr) {
  if (!isActive()) {
    return;
  }
  state_->op_input_size = state_->inputs_.size();
  state_->name_ = name;
  state_->sequence_nr_ = sequence_nr;
  state_->thread_id_ = currentThreadId();
  state_->operator_name_.reset();

  manager().runStartCallbacks(*this);
  invalidateInputs();
}

void RecordFunction::before(std::string name, int64_t sequence_nr) {
  if (!isActive()) {
    return;
  }
  state_->op_input_size = state_->inputs_.size();
  state_->name_ = std::move(name);
  state_->sequence_nr_ = sequence_nr;
  state_->thread_id_ = currentThreadId();
  state_->operator_name_.reset();

  manager().runStartCallbacks(*this);
  invalidateInputs();
}

void RecordFunction::before(
    c10::OperatorHandle const& op,
    int64_t sequence_nr) {
  if (!isActive()) {
    return;
  }
  state_->sequence_nr_ = sequence_nr;
  state_->thread_id_ = currentThreadId();
  state_->operator_name_ = op.operator_name();
  state_->op_input_size = op.schema().arguments().size();
  state_->op_output_size = op.schema().returns().size();
  state_->name_ = op.schema().name();

  manager().runStartCallbacks(*this);
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

void RecordFunction::end() {
  if (isActive() && state_->called_start_callbacks_) {
    manager().runEndCallbacks(*this);
    state_.reset();
  }
}

void RecordFunction::_setAsync() {
  if (isActive()) {
    state_->is_async_ = true;
  }
}

bool RecordFunction::isAsync() const {
  if (isActive()) {
    return state_->is_async_;
  }
  return false;
}

// RecordFunction pre-sampling
namespace {
// Whether to try to create RecordFunction on each call (>0) or
// use pre-sampling (=0)
std::atomic<int> global_record_all_functions_ {0};
}

void bumpRecordAllFunctions() {
  global_record_all_functions_.fetch_add(1, std::memory_order_relaxed);
}

void releaseRecordAllFunctions() {
  TORCH_CHECK(global_record_all_functions_.fetch_sub(1, std::memory_order_relaxed) > 0);
}

bool checkRecordAllFunctions() {
  return (global_record_all_functions_.load(std::memory_order_relaxed) > 0);
}

bool shouldRunRecordFunction(bool* pre_sampled) {
  auto* rf_tls_ptr = &rf_tls();
  if (rf_tls_ptr->sorted_tls_callbacks_.empty() && !manager().hasGlobalCallbacks()) {
    *pre_sampled = false;
    return false;
  }
  if (global_record_all_functions_.load(std::memory_order_relaxed) > 0) {
    *pre_sampled = false;
    return true;
  }
  if (!rf_tls_ptr->tls_record_function_enabled_) {
    *pre_sampled = false;
    return false;
  }

  *pre_sampled = true;
  auto* coinflip_tls_ptr = &coinflip_tls();
  if (coinflip_tls_ptr->tries_left_ == 0) {
    coinflip_tls_ptr->tries_left_ = sample_geometric();
    return true;
  } else {
    --coinflip_tls_ptr->tries_left_;
    return false;
  }
}

} // namespace at
