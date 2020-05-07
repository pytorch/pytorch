#include <ATen/record_function.h>
#include <algorithm>
#include <cstdlib>
#include <random>

namespace at {

namespace {

// Used to generate unique callback handles
CallbackHandle next_unique_callback_handle() {
  static std::atomic<uint64_t> unique_id {0};
  return CallbackHandle(++unique_id);
}

// Thread local vector of callbacks, holds pairs (callbacks, unique_id);
// must be sorted in increasing handles order
thread_local RecordFunctionCallbacks sorted_tls_callbacks_;

class CallbackManager {
 public:
  CallbackHandle addThreadLocalCallback(RecordFunctionCallback cb) {
    // note: monotonically increasing callbacks_unique_id keeps
    // sorted_tls_callbacks_ sorted
    auto handle = next_unique_callback_handle();
    sorted_tls_callbacks_.emplace_back(std::move(cb), handle);
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
    auto found = find_and_remove(sorted_tls_callbacks_);
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
    sorted_tls_callbacks_.clear();
  }

  inline bool hasGlobalCallbacks() const {
    return !sorted_global_callbacks_.empty();
  }

  inline bool hasThreadLocalCallbacks() const {
    return !sorted_tls_callbacks_.empty();
  }

  // init is called by RecordFunction in constructor to
  // determine which thread local and global callbacks are going
  // to be executed and whether any of them need inputs
  inline void init(RecordFunction& rec_fn) {
    auto scope = rec_fn.scope();
    bool found_active_cb = false;
    bool found_needs_inputs = false;
    auto init_handles = [scope, &found_active_cb, &found_needs_inputs](
        CallbackHandles& handles, RecordFunctionCallbacks& cbs) {
      handles.clear();
      for (const auto& cb : cbs) {
        if (cb.first.shouldRun(scope)) {
          handles.push_back(cb.second);
          found_active_cb = true;
          if (cb.first.needsInputs()) {
            found_needs_inputs = true;
          }
        }
      }
    };

    init_handles(rec_fn.sorted_active_tls_handles_, sorted_tls_callbacks_);
    init_handles(rec_fn.sorted_active_global_handles_, sorted_global_callbacks_);
    rec_fn.active_ = found_active_cb;
    rec_fn.needs_inputs_ = found_needs_inputs;
  }

  void runStartCallbacks(RecordFunction& rf) {
    mergeRunCallbacks(
        sorted_global_callbacks_,
        rf.sorted_active_global_handles_,
        /* is_start */ true,
        rf);
    mergeRunCallbacks(
        sorted_tls_callbacks_,
        rf.sorted_active_tls_handles_,
        /* is_start */ true,
        rf);
  }

  void runEndCallbacks(RecordFunction& rf) {
    mergeRunCallbacks(
        sorted_global_callbacks_,
        rf.sorted_active_global_handles_,
        /* is_start */ false,
        rf);
    mergeRunCallbacks(
        sorted_tls_callbacks_,
        rf.sorted_active_tls_handles_,
        /* is_start */ false,
        rf);
  }

 private:
  bool tryRunCallback(
      const std::function<void(const RecordFunction&)>& fn,
      RecordFunction& rf) {
    try {
      fn(rf);
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
      bool is_start,
      RecordFunction& rf) {
    size_t num_executed = 0;
    size_t idx_c = 0;
    for (size_t idx_h = 0; idx_h < sorted_handles.size(); ++idx_h) {
      while (idx_c < sorted_callbacks.size() &&
            sorted_callbacks[idx_c].second < sorted_handles[idx_h]) {
        ++idx_c;
      }
      if (idx_c >= sorted_callbacks.size()) {
        break;
      }
      if (sorted_callbacks[idx_c].second == sorted_handles[idx_h]) {
        if (is_start) {
          tryRunCallback(sorted_callbacks[idx_c].first.start(), rf);
        } else {
          tryRunCallback(sorted_callbacks[idx_c].first.end(), rf);
        }
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

  // Global callbacks; must be sorted in increasing handle order
  RecordFunctionCallbacks sorted_global_callbacks_;
};

// Enumerates thread ids logically;
// note: std::this_thread::get_id may return potentially
// reused thread id
std::atomic<uint64_t> next_thread_id_ {0};
thread_local uint64_t current_thread_id_ = 0;

// Points to the currently active RecordFunction
thread_local RecordFunction* current_record_func_ = nullptr;

inline CallbackManager& manager() {
  static CallbackManager _manager;
  return _manager;
}

thread_local bool tls_record_function_enabled_ = true;

} // namespace

/* static */
double RecordFunctionCallback::sample_zero_one() {
  static thread_local auto gen =
      std::make_unique<std::mt19937>(std::random_device()());
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(*gen);
}

RecordFunctionCallbacks _getTLSCallbacks() {
  return sorted_tls_callbacks_;
}

void _setTLSCallbacks(const RecordFunctionCallbacks& callbacks) {
  // keep the original handles
  sorted_tls_callbacks_ = callbacks;
  std::sort(
      sorted_tls_callbacks_.begin(),
      sorted_tls_callbacks_.end(),
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
  return tls_record_function_enabled_;
}

void enableRecordFunction(bool enable) {
  tls_record_function_enabled_ = enable;
}

RecordFunction::RecordFunction(RecordScope scope) : scope_(scope) {
  if (hasCallbacks() && isRecordFunctionEnabled()) {
    manager().init(*this);
  }
}

void RecordFunction::_setCurrent() {
  parent_ = current_record_func_;
  current_record_func_ = this;
  is_current_ = true;
}

/* static */
uint64_t RecordFunction::currentThreadId() {
  if (!current_thread_id_) {
    // happens only once per thread
    current_thread_id_ = ++next_thread_id_;
  }
  return current_thread_id_;
}

void RecordFunction::_before(const char* name, int64_t sequence_nr) {
  if (!active_) {
    return;
  }
  name_ = StringView(name);
  sequence_nr_ = sequence_nr;
  thread_id_ = currentThreadId();

  manager().runStartCallbacks(*this);
}

void RecordFunction::_before(std::string name, int64_t sequence_nr) {
  if (!active_) {
    return;
  }
  name_ = StringView(std::move(name));
  sequence_nr_ = sequence_nr;
  thread_id_ = currentThreadId();

  manager().runStartCallbacks(*this);
}

RecordFunction::~RecordFunction() {
  _end();
}

void RecordFunction::_end() {
  if (active_) {
    manager().runEndCallbacks(*this);
    active_ = false;
  }
  if (is_current_) {
    current_record_func_ = parent_;
    is_current_ = false;
  }
}

/* static */
RecordFunction* RecordFunction::current() {
  return current_record_func_;
}

} // namespace at
