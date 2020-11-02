#include <ATen/record_function.h>
#include <ATen/core/dispatch/Dispatcher.h>
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

thread_local RecordFunctionTLS rf_tls_;

std::atomic<int64_t> defaultNodeId(-1);

// Enumerates thread ids logically;
// note: std::this_thread::get_id may return potentially
// reused thread id
std::atomic<uint64_t> next_thread_id_ {0};
thread_local uint64_t current_thread_id_ = 0;

thread_local bool tls_record_function_enabled_ = true;

// Low probability constant
const double kLowProb = 0.001;
thread_local int tries_left_ = 0;

int sample_geometric() {
  static thread_local auto gen =
      std::make_unique<std::mt19937>(std::random_device()());
  std::geometric_distribution<int> dist(kLowProb);
  return dist(*gen);
}

double sample_zero_one() {
  static thread_local auto gen =
      std::make_unique<std::mt19937>(std::random_device()());
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(*gen);
}

} // namespace

const RecordFunctionTLS& get_record_function_tls_() {
  return rf_tls_;
}

void set_record_function_tls_(const RecordFunctionTLS& tls) {
  rf_tls_ = tls;
}

class CallbackManager {
 public:
  CallbackHandle addThreadLocalCallback(RecordFunctionCallback cb) {
    // note: monotonically increasing callbacks_unique_id keeps
    // sorted_tls_callbacks_ sorted
    auto handle = next_unique_callback_handle();
    rf_tls_.sorted_tls_callbacks_.emplace_back(std::move(cb), handle);
    return handle;
  }

  CallbackHandle addGlobalCallback(RecordFunctionCallback cb) {
    auto handle = next_unique_callback_handle();
    sorted_global_callbacks_.emplace_back(std::move(cb), handle);
    return handle;
  }

  void removeCallback(CallbackHandle handle) {
    auto find_and_remove = [handle](RecordFunctionCallbacks& cbs) {
      auto it = std::find_if(
        cbs.begin(), cbs.end(),
        [handle](
            const std::pair<
                RecordFunctionCallback,
                CallbackHandle>& el) {
          return el.second == handle;
        });
      if (it != cbs.end()) {
        // keeps it sorted
        cbs.erase(it);
        return true;
      }
      return false;
    };
    auto found = find_and_remove(rf_tls_.sorted_tls_callbacks_);
    if (!found) {
      found = find_and_remove(sorted_global_callbacks_);
    }
    if (!found) {
      LOG(WARNING) << "Requested callback is not found";
    }
  }

  void clearGlobalCallbacks() {
    sorted_global_callbacks_.clear();
  }

  void clearThreadLocalCallbacks() {
    rf_tls_.sorted_tls_callbacks_.clear();
  }

  inline bool hasGlobalCallbacks() const {
    return !sorted_global_callbacks_.empty();
  }

  inline bool hasThreadLocalCallbacks() const {
    return !rf_tls_.sorted_tls_callbacks_.empty();
  }

  // init is called by RecordFunction in constructor to
  // determine which thread local and global callbacks are going
  // to be executed and whether any of them need inputs
  inline void init(RecordFunction& rec_fn) {
    auto scope = rec_fn.scope();
    bool found_active_cb = false;
    bool found_needs_inputs = false;
    bool found_needs_ids = false;
    auto init_handles = [
        scope, &found_active_cb, &found_needs_inputs, &found_needs_ids](
          CallbackHandles& handles, RecordFunctionCallbacks& cbs, ObserverContextList& ctx_list) {
      handles.clear();

      size_t num_callbacks = 0;
      for (const auto& cb : cbs) {
        if (cb.first.shouldRun(scope)) {
          handles.push_back(cb.second);
          ++num_callbacks;
          found_active_cb = true;
          if (cb.first.needsInputs()) {
            found_needs_inputs = true;
          }
          if (cb.first.needsIds()) {
            found_needs_ids = true;
          }
        }
      }
      // Pre-allocate observer context list with nullptr.
      ctx_list.resize(num_callbacks);
    };

    init_handles(rec_fn.sorted_active_tls_handles_, rf_tls_.sorted_tls_callbacks_, rec_fn.tls_ctx_);
    init_handles(rec_fn.sorted_active_global_handles_, sorted_global_callbacks_, rec_fn.global_ctx_);
    rec_fn.active = found_active_cb;
    rec_fn.needs_inputs = found_needs_inputs;
    if (found_needs_ids && found_active_cb) {
      rec_fn.setHandle(next_unique_record_function_handle());
    }
  }

  void runStartCallbacks(RecordFunction& rf) {
    mergeRunCallbacks(
        sorted_global_callbacks_,
        rf.sorted_active_global_handles_,
        rf.global_ctx_,
        /* is_start */ true,
        rf);
    mergeRunCallbacks(
        rf_tls_.sorted_tls_callbacks_,
        rf.sorted_active_tls_handles_,
        rf.tls_ctx_,
        /* is_start */ true,
        rf);
    rf.called_start_callbacks_ = true;
  }

  void runEndCallbacks(RecordFunction& rf) {
    mergeRunCallbacks(
        sorted_global_callbacks_,
        rf.sorted_active_global_handles_,
        rf.global_ctx_,
        /* is_start */ false,
        rf);
    mergeRunCallbacks(
        rf_tls_.sorted_tls_callbacks_,
        rf.sorted_active_tls_handles_,
        rf.tls_ctx_,
        /* is_start */ false,
        rf);
  }

  // Global callbacks; must be sorted in increasing handle order
  RecordFunctionCallbacks sorted_global_callbacks_;

 private:
  bool tryRunCallback(
      const RecordFunctionCallback& rfcb,
      RecordFunction& rf,
      std::unique_ptr<ObserverContext>& ctx,
      bool is_start) {
    try {
      if (is_start) {
        ctx = rfcb.start()(rf);
      }
      else {
        rfcb.end()(rf, ctx.get());
      }
      return true;
    } catch (const std::exception &e) {
      LOG(WARNING) << "Exception in RecordFunction callback: "
          << e.what() << " , for the range " << rf.name();
      return false;
    } catch (...) {
      LOG(WARNING) << "Exception in RecordFunction callback: unknown"
          << " , for the range " << rf.name();
      return false;
    }
  }

  void mergeRunCallbacks(
      const RecordFunctionCallbacks& sorted_callbacks,
      const CallbackHandles& sorted_handles,
      ObserverContextList& ctx_list,
      bool is_start,
      RecordFunction& rf) {
    size_t num_executed = 0;
    size_t idx_c = 0;
    for (size_t idx_h = 0; idx_h < sorted_handles.size() && idx_h < ctx_list.size(); ++idx_h) {
      while (idx_c < sorted_callbacks.size() &&
            sorted_callbacks[idx_c].second < sorted_handles[idx_h]) {
        ++idx_c;
      }
      if (idx_c >= sorted_callbacks.size()) {
        break;
      }
      if (sorted_callbacks[idx_c].second == sorted_handles[idx_h]) {
        tryRunCallback(sorted_callbacks[idx_c].first, rf, ctx_list[idx_h], is_start);
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

bool RecordFunctionCallback::shouldRun(RecordScope scope) const {
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
    // model the low probability events as events happening
    // with prob. kLowProb followed by another sampling with
    // prob. (sampling_prob_ / kLowProb), then replace the coin
    // flip for kLowProb with a thread local number of tries tries_left_
    // sampled from the geometric distribution
    if (sampling_prob_ < kLowProb) {
      if (tries_left_ == 0) {
        tries_left_ = sample_geometric();
        return (sample_zero_one() < sampling_prob_ / kLowProb);
      } else {
        --tries_left_;
        return false;
      }
    } else {
      return (sample_zero_one() < sampling_prob_);
    }
  }
  return true;
}

RecordFunctionCallbacks _getTLSCallbacks() {
  return rf_tls_.sorted_tls_callbacks_;
}

void _setTLSCallbacks(const RecordFunctionCallbacks& callbacks) {
  // keep the original handles
  rf_tls_.sorted_tls_callbacks_ = callbacks;
  std::sort(
      rf_tls_.sorted_tls_callbacks_.begin(),
      rf_tls_.sorted_tls_callbacks_.end(),
      [](const std::pair<RecordFunctionCallback, CallbackHandle>& l,
          const std::pair<RecordFunctionCallback, CallbackHandle>& r) {
        return l.second < r.second;
  });
}

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
  return manager().addThreadLocalCallback(std::move(cb));
}

CallbackHandle addGlobalCallback(
    RecordFunctionCallback cb) {
  return manager().addGlobalCallback(std::move(cb));
}

void removeCallback(CallbackHandle handle) {
  manager().removeCallback(handle);
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
  return rf_tls_.tls_record_function_enabled_;
}

void enableRecordFunction(bool enable) {
  rf_tls_.tls_record_function_enabled_ = enable;
}

RecordFunction::RecordFunction(RecordScope scope) : scope_(scope) {
  auto* rf_tls_ptr = &rf_tls_;
  auto& m = manager();
  if (rf_tls_ptr->tls_record_function_enabled_ &&
      (!m.sorted_global_callbacks_.empty() || !rf_tls_ptr->sorted_tls_callbacks_.empty())) {
    m.init(*this);
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
  if (!active) {
    return;
  }
  name_ = StringView(name);
  sequence_nr_ = sequence_nr;
  thread_id_ = currentThreadId();
  operator_name_.reset();

  manager().runStartCallbacks(*this);
}

void RecordFunction::before(std::string name, int64_t sequence_nr) {
  if (!active) {
    return;
  }
  name_ = StringView(std::move(name));
  sequence_nr_ = sequence_nr;
  thread_id_ = currentThreadId();
  operator_name_.reset();

  manager().runStartCallbacks(*this);
}

void RecordFunction::before(
    c10::OperatorHandle const& op,
    int64_t sequence_nr) {
  if (!active) {
    return;
  }
  sequence_nr_ = sequence_nr;
  thread_id_ = currentThreadId();
  operator_name_ = op.operator_name();
  name_ = StringView(op.schema().name());

  manager().runStartCallbacks(*this);
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
  if (active && called_start_callbacks_) {
    manager().runEndCallbacks(*this);
  }
  active = false;
}

} // namespace at
