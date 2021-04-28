#include <ATen/record_function.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/macros/Macros.h>
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

// Low probability constant
static const double kLowProb = 0.001;
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
thread_local CoinflipTLS coinflip_tls_;

int sample_geometric() {
  return coinflip_tls_.distGeo_(coinflip_tls_.genGeo_);
}

double sample_zero_one() {
  return coinflip_tls_.distZeroOne_(coinflip_tls_.genZeroOne_);
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
    if (cb.samplingProb() > kLowProb) {
      // pre-sampling of RecordFunction with prob. kLowProb cannot be used
      at::bumpRecordAllFunctions();
    }
    // note: monotonically increasing callbacks_unique_id keeps
    // sorted_tls_callbacks_ sorted
    auto handle = next_unique_callback_handle();
    rf_tls_.sorted_tls_callbacks_.emplace_back(std::move(cb), handle);
    return handle;
  }

  CallbackHandle addGlobalCallback(RecordFunctionCallback cb) {
    if (cb.samplingProb() > kLowProb) {
      // pre-sampling of RecordFunction with prob. kLowProb cannot be used
      at::bumpRecordAllFunctions();
    }
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
        if (it->first.samplingProb() > kLowProb) {
          // try to restore pre-sampling of RecordFunction
          at::releaseRecordAllFunctions();
        }
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
    // if we have registered should_run_ function, use it
    if (cb.should_run_) {
      return cb.should_run_(cb);
    }

    // otherwise potentially do the sampling
    double sampling_prob = cb.sampling_prob_;
    if (pre_sampled) {
      // adjust the sampling rate to account for kLowProb pre-sampling of
      // the RecordFunction
      sampling_prob /= kLowProb;
    }

    if (sampling_prob < 1.0) {
      // model the low probability events as events happening
      // with probability kLowProb followed by another sampling with
      // probability (sampling_prob / kLowProb), then replace the coin
      // flip for kLowProb with a thread local number of tries tries_left_
      // sampled from the geometric distribution.
      if (sampling_prob < kLowProb) {
        if (coinflip_tls_.tries_left_ == 0) {
          coinflip_tls_.tries_left_ = sample_geometric();
          return (sample_zero_one() < sampling_prob / kLowProb);
        } else {
          --coinflip_tls_.tries_left_;
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

    for (const auto& cb: rf_tls_.sorted_tls_callbacks_) {
      if (callbackShouldRun(cb.first, scope, pre_sampled)) {
        if (cb.first.needsInputs()) {
          found_needs_inputs = true;
        }
        if (cb.first.needsOutputs()) {
          found_needs_outputs = true;
        }
        if (cb.first.needsIds()) {
          found_needs_ids = true;
        }
        if (!rec_fn.state_) {
          rec_fn.state_ = std::make_unique<RecordFunction::State>(scope);
        }
        rec_fn.state_->sorted_active_tls_handles_.push_back(cb.second);
      }
    }

    for (const auto& cb: sorted_global_callbacks_) {
      if (callbackShouldRun(cb.first, scope, pre_sampled)) {
        if (cb.first.needsInputs()) {
          found_needs_inputs = true;
        }
        if (cb.first.needsOutputs()) {
          found_needs_outputs = true;
        }
        if (cb.first.needsIds()) {
          found_needs_ids = true;
        }
        if (!rec_fn.state_) {
          rec_fn.state_ = std::make_unique<RecordFunction::State>(scope);
        }
        rec_fn.state_->sorted_active_global_handles_.push_back(cb.second);
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
        rf_tls_.sorted_tls_callbacks_,
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
        rf_tls_.sorted_tls_callbacks_,
        rf.state_->sorted_active_tls_handles_,
        rf.state_->tls_ctx_,
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
        ctx = rfcb.start() ? rfcb.start()(rf) : nullptr;
      }
      else {
        if (rfcb.end()) {
          rfcb.end()(rf, ctx.get());
        }
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

RecordFunction::RecordFunction(RecordScope scope, bool pre_sampled) {
  auto* rf_tls_ptr = &rf_tls_;
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
  state_->name_ = StringView(name);
  state_->sequence_nr_ = sequence_nr;
  state_->thread_id_ = currentThreadId();
  state_->operator_name_.reset();

  manager().runStartCallbacks(*this);
}

void RecordFunction::before(std::string name, int64_t sequence_nr) {
  if (!isActive()) {
    return;
  }
  state_->name_ = StringView(std::move(name));
  state_->sequence_nr_ = sequence_nr;
  state_->thread_id_ = currentThreadId();
  state_->operator_name_.reset();

  manager().runStartCallbacks(*this);
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
  state_->name_ = StringView(op.schema().name());

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
  if (isActive() && state_->called_start_callbacks_) {
    manager().runEndCallbacks(*this);
    state_.reset();
  }
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
  TORCH_CHECK(global_record_all_functions_.fetch_sub(1, std::memory_order_relaxed) >= 0);
}

bool checkRecordAllFunctions() {
  return (global_record_all_functions_.load(std::memory_order_relaxed) > 0);
}

bool shouldRunRecordFunction(bool* pre_sampled) {
  auto* rf_tls_ptr = &rf_tls_;
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
  auto* coinflip_tls_ptr = &coinflip_tls_;
  if (coinflip_tls_ptr->tries_left_ == 0) {
    coinflip_tls_ptr->tries_left_ = sample_geometric();
    return true;
  } else {
    --coinflip_tls_ptr->tries_left_;
    return false;
  }
}

} // namespace at
