
#include <algorithm>
#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/profiler.h>
#include <cstdlib>
#include <random>

namespace torch {
namespace autograd {
namespace profiler {

namespace {

// Used to generate unique callback handles
uint64_t next_unique_callback_id() {
  static std::mutex id_mutex;
  static uint64_t unique_id = 0;
  std::lock_guard<std::mutex> id_lock(id_mutex);
  return ++unique_id;
}
// Thread local vector of callbacks, holds pairs (callbacks, unique_id);
// must be sorted in increase handles order
typedef std::vector<std::pair<RecordFunctionCallback, uint64_t>> idd_callbacks_vector;
thread_local idd_callbacks_vector sorted_tls_callbacks_;

class CallbackManager {
 public:
  uint64_t addThreadLocalCallback(RecordFunctionCallback cb) {
    // note: monotonically increasing callbacks_unique_id keeps
    // sorted_tls_callbacks_ sorted
    auto handle = next_unique_callback_id();
    sorted_tls_callbacks_.emplace_back(std::move(cb), handle);
    return handle;
  }

  uint64_t addGlobalCallback(RecordFunctionCallback cb) {
    auto handle = next_unique_callback_id();
    sorted_global_callbacks_.emplace_back(std::move(cb), handle);
    return handle;
  }

  void removeCallback(uint64_t handle) {
    bool found = false;
    auto find_and_remove = [handle, &found](idd_callbacks_vector& cbs) {
      auto it = std::find_if(
        cbs.begin(), cbs.end(),
        [handle](const std::pair<RecordFunctionCallback, uint64_t>& el) {
          return el.second == handle;
        });
      if (it != cbs.end()) {
        // keeps it sorted
        cbs.erase(it);
      }
    };
    find_and_remove(sorted_tls_callbacks_);
    if (!found) {
      find_and_remove(sorted_global_callbacks_);
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
        handles_vector& handles, idd_callbacks_vector& cbs) {
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
  bool tryRunCallback(const std::function<void(const RecordFunction&)>& fn, RecordFunction& rf) {
    try {
      fn(rf);
      return true;
    } catch (const std::exception &e) {
      LOG(WARNING) << "Exception in RecordFunction callback: "
                    << e.what();
      return false;
    } catch (...) {
      LOG(WARNING) << "Exception in RecordFunction callback: unknown";
      return false;
    }
  }

  void mergeRunCallbacks(
      const idd_callbacks_vector& sorted_callbacks,
      handles_vector sorted_handles,
      bool is_start,
      RecordFunction& rf) {
    size_t idx_c = 0;
    size_t idx_h = 0;
    size_t num_executed = 0;
    while (idx_c < sorted_callbacks.size() && idx_h < sorted_handles.size()) {
      if (sorted_callbacks[idx_c].second == sorted_handles[idx_h]) {
        if (is_start) {
          tryRunCallback(sorted_callbacks[idx_c].first.start(), rf);
        } else {
          tryRunCallback(sorted_callbacks[idx_c].first.end(), rf);
        }
        ++num_executed;
        ++idx_c;
        ++idx_h;
      }
      while ((sorted_callbacks[idx_c].second < sorted_handles[idx_h])
          && (idx_c < sorted_callbacks.size())) {
        ++idx_c;
      }
      while ((sorted_handles[idx_h] < sorted_callbacks[idx_c].second)
          && (idx_h < sorted_handles.size())) {
        ++idx_h;
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
  idd_callbacks_vector sorted_global_callbacks_;
};

// Enumerates thread ids logically;
// note: std::this_thread::get_id may return potentially
// reused thread id
std::mutex next_thread_id_mutex_;
uint16_t next_thread_id_ = 0;
thread_local uint16_t current_thread_id_ = 0;

// Points to the currently active RecordFunction
thread_local RecordFunction* current_record_func_ = nullptr;

inline CallbackManager& manager() {
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

uint64_t addThreadLocalCallback(RecordFunctionCallback cb) {
  return manager().addThreadLocalCallback(std::move(cb));
}

uint64_t addGlobalCallback(RecordFunctionCallback cb) {
  return manager().addGlobalCallback(std::move(cb));
}

void removeCallback(uint64_t handle) {
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
  return c10::impl::tls_is_dispatch_key_included(c10::DispatchKey::Profiler);
}

void enableRecordFunction(bool enable) {
  c10::impl::tls_set_dispatch_key_included(c10::DispatchKey::Profiler, enable);
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
uint16_t RecordFunction::currentThreadId() {
  if (!current_thread_id_) {
    // happens only once per thread
    std::lock_guard<std::mutex> guard(next_thread_id_mutex_);
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

void RecordFunction::_before(Node* fn, int64_t sequence_nr) {
  if (!active_) {
    return;
  }
  fn_ = fn;
  name_ = StringView(fn->name());
  sequence_nr_ = (sequence_nr >= 0) ? sequence_nr : fn->sequence_nr();
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

} // namespace profiler
} // namespace autograd
} // namespace torch
