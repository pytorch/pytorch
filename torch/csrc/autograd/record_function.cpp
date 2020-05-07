#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/utils/memory.h>
#include <cstdlib>
#include <random>

namespace torch {
namespace autograd {
namespace profiler {

namespace {

float sample_zero_one() {
  static thread_local auto gen =
      torch::make_unique<std::mt19937>(std::random_device()());
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  return dist(*gen);
}

class CallbackManager {
 public:
  void pushCallback(
      std::function<bool(const RecordFunction&)> start,
      std::function<void(const RecordFunction&)> end,
      bool needs_inputs,
      double sampling_prob,
      std::unordered_set<RecordScope, std::hash<RecordScope>> scopes) {
    callbacks_.emplace_back(
      std::move(start),
      std::move(end),
      needs_inputs,
      sampling_prob,
      std::move(scopes)
    );
    recomputeFlags();

    // make sure we mark the change in callbacks
    ++callbacks_version_;
  }

  void popCallback() {
    if (callbacks_.empty()) {
      throw std::runtime_error("Empty callbacks stack");
    }
    callbacks_.pop_back();
    recomputeFlags();
    ++callbacks_version_;
  }

  inline bool hasCallbacks() const {
    return !callbacks_.empty();
  }

  inline bool needsInputs() const {
    return has_callbacks_with_inputs_;
  }

  void runStartCallbacks(RecordFunction& rf) {
    rf._setCallbacksVersion(callbacks_version_);
    rf._activeCallbacks().clear();
    for (size_t cb_idx = 0; cb_idx < callbacks_.size(); ++cb_idx) {
      if (shouldRunCallback(cb_idx, rf.scope())) {
        try {
          bool cb_ret = callbacks_[cb_idx].start_cb_(rf);
          rf._activeCallbacks().push_back(cb_ret);
        } catch (const std::exception &e) {
          LOG(WARNING) << "Exception in RecordFunction start observer: "
                       << e.what();
          rf._activeCallbacks().push_back(false);
        } catch (...) {
          LOG(WARNING) << "Exception in RecordFunction start observer: unknown";
          rf._activeCallbacks().push_back(false);
        }
      } else {
        rf._activeCallbacks().push_back(false);
      }
    }
  }

  void runEndCallbacks(RecordFunction& rf) {
    if (rf._callbacksVersion() == callbacks_version_) {
      for (size_t cb_idx = 0; cb_idx < rf._activeCallbacks().size(); ++cb_idx) {
        if (!rf._activeCallbacks()[cb_idx]) {
          continue;
        }
        try {
          callbacks_[cb_idx].end_cb_(rf);
        } catch (const std::exception &e) {
          LOG(WARNING) << "Exception in RecordFunction end observer: "
                       << e.what();
        } catch (...) {
          LOG(WARNING) << "Exception in RecordFunction end observer: unknown";
        }
      }
    } else {
      LOG(WARNING) << "Callbacks changed while running a record function, "
                   << "you might be partially overlapping a record function "
                   << "with a profiling scope";
    }
  }

  inline void TEST_setGlobalSamplingProbability(double sampling_prob) {
    global_prob_ = sampling_prob;
    use_global_prob_ = true;
  }

  inline void TEST_unsetGlobalSamplingProbability() {
    global_prob_ = 0.0;
    use_global_prob_ = false;
  }

 private:
  void recomputeFlags() {
    has_callbacks_with_inputs_ = false;
    for (const auto& cb : callbacks_) {
      has_callbacks_with_inputs_ |= cb.needs_inputs_;
    }
  }

  inline double samplingProbability(size_t cb_idx) const {
    TORCH_INTERNAL_ASSERT(cb_idx < callbacks_.size());
    if (callbacks_[cb_idx].is_sampled_) {
      return use_global_prob_ ? global_prob_ : callbacks_[cb_idx].sampling_prob_;
    } else {
      return 1.0;
    }
  }

  inline bool shouldRunCallback(size_t cb_idx, RecordScope scope) const {
    TORCH_INTERNAL_ASSERT(cb_idx < callbacks_.size());
    return callbacks_[cb_idx].scopes_[static_cast<size_t>(scope)] &&
           (!callbacks_[cb_idx].is_sampled_ ||
            (sample_zero_one() < samplingProbability(cb_idx)));
  }

  struct Callback;
  std::vector<Callback> callbacks_;

  double global_prob_ = 0.0;
  bool use_global_prob_ = false;
  bool has_callbacks_with_inputs_ = false;

  // tracks the current 'version' of callbacks;
  // every time we push or pop callbacks, we bump this counter
  uint64_t callbacks_version_ = 0;

  struct Callback {
    Callback(
        std::function<bool(const RecordFunction&)> start_cb,
        std::function<void(const RecordFunction&)> end_cb,
        bool needs_inputs,
        double sampling_prob,
        std::unordered_set<RecordScope, std::hash<RecordScope>> scopes
    ) : start_cb_(std::move(start_cb)),
        end_cb_(std::move(end_cb)),
        needs_inputs_(needs_inputs),
        sampling_prob_(sampling_prob),
        is_sampled_(sampling_prob != 1.0) {
      if (!scopes.empty()) {
        scopes_.fill(false);
        for (auto sc : scopes) {
          scopes_[static_cast<size_t>(sc)] = true;
        }
      } else {
        scopes_.fill(true);
      }
    }

    std::function<bool(const RecordFunction&)> start_cb_;
    std::function<void(const RecordFunction&)> end_cb_;
    std::array<bool, static_cast<size_t>(RecordScope::NUM_SCOPES)> scopes_;
    const bool needs_inputs_;
    const double sampling_prob_;
    const bool is_sampled_;
  };
};

std::mutex next_thread_id_mutex_;
uint16_t next_thread_id_ = 0;
thread_local uint16_t current_thread_id_ = 0;

// points to the currently active RecordFunction
thread_local RecordFunction* current_record_func_ = nullptr;

inline CallbackManager& manager() {
  static CallbackManager _manager;
  return _manager;
}

} // namespace

bool hasCallbacks() {
  return manager().hasCallbacks();
}

void pushCallback(
    std::function<bool(const RecordFunction&)> start,
    std::function<void(const RecordFunction&)> end,
    bool needs_inputs,
    double sampling_prob,
    std::unordered_set<RecordScope, std::hash<RecordScope>> scopes) {
  manager().pushCallback(
      std::move(start),
      std::move(end),
      needs_inputs,
      sampling_prob,
      std::move(scopes));
}

void popCallback() {
  manager().popCallback();
}

void _runBeforeCallbacks(RecordFunction* rf, const std::string& funcName) {
  TORCH_INTERNAL_ASSERT(rf != nullptr);
  rf->_before(funcName);
}

RecordFunction::RecordFunction(RecordScope scope) : scope_(scope) {
  if (manager().hasCallbacks()) {
    active_ = true;
  }
}

void RecordFunction::_setCurrent() {
  parent_ = current_record_func_;
  current_record_func_ = this;
  is_current_ = true;
}

/* static */
bool RecordFunction::_needsInputs() {
  return manager().needsInputs();
}

void TEST_setGlobalSamplingProbability(double sampling_prob) {
  manager().TEST_setGlobalSamplingProbability(sampling_prob);
}

void TEST_unsetGlobalSamplingProbability() {
  manager().TEST_unsetGlobalSamplingProbability();
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

  processCallbacks();
}

void RecordFunction::_before(std::string name, int64_t sequence_nr) {
  if (!active_) {
    return;
  }
  name_ = StringView(std::move(name));
  sequence_nr_ = sequence_nr;

  processCallbacks();
}

void RecordFunction::_before(Node* fn, int64_t sequence_nr) {
  if (!active_) {
    return;
  }
  fn_ = fn;
  name_ = StringView(fn->name());
  sequence_nr_ = (sequence_nr >= 0) ? sequence_nr : fn->sequence_nr();

  processCallbacks();
}

void RecordFunction::processCallbacks() {
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
