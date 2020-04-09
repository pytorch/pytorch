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

class RecordFunctionCallback {
 public:
  RecordFunctionCallback(
      std::function<void(const RecordFunction&)> start,
      std::function<void(const RecordFunction&)> end =
        [](const RecordFunction&) {}):
      start_(std::move(start)),
      end_(std::move(end)) {}

  RecordFunctionCallback& needsInputs(bool needs_inputs) {
    needs_inputs_ = needs_inputs;
    return *this;
  }

  RecordFunctionCallback& samplingProb(double sampling_prob) {
    sampling_prob_ = sampling_prob;
    is_sampled_ = sampling_prob_ != 1.0;
    return *this;
  }

  RecordFunctionCallback& scopes(
      std::unordered_set<RecordScope, std::hash<RecordScope>> scopes) {
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

  inline bool needsInputs() const {
    return needs_inputs_;
  }

  inline double samplingProb() const {
    return sampling_prob_;
  }

  inline bool checkScope(RecordScope sc) const {
    return scopes_[(size_t)sc];
  }

  inline bool isSampled() const {
    return is_sampled_;
  }

  inline std::function<void(const RecordFunction&)>& start() {
    return start_;
  }

  inline std::function<void(const RecordFunction&)>& end() {
    return end_;
  }

 private:
  std::function<void(const RecordFunction&)> start_;
  std::function<void(const RecordFunction&)> end_;
  bool needs_inputs_ = false;
  double sampling_prob_ = 1.0;
  bool is_sampled_ = false;
  std::array<bool, static_cast<size_t>(RecordScope::NUM_SCOPES)> scopes_;
};

thread_local std::vector<RecordFunctionCallback> tls_callbacks_;

class CallbackManager {
 public:
  void pushCallback(RecordFunctionCallback cb, bool is_thread_local) {
    TORCH_CHECK(
        !is_thread_local || !cb.isSampled(),
        "Sampled thread local callbacks are not supported");
    auto& cb_list = is_thread_local ? tls_callbacks_ : callbacks_;
    cb_list.emplace_back(std::move(cb));
    if (!is_thread_local) {
      ++global_callbacks_version_;
    }
    recomputeFlags();
  }

  void popCallback(bool is_thread_local) {
    auto& cb_list = is_thread_local ? tls_callbacks_ : callbacks_;
    TORCH_CHECK(!cb_list.empty(), "Empty callbacks stack");
    cb_list.pop_back();
    if (!is_thread_local) {
      ++global_callbacks_version_;
    }
  }

  inline bool hasGlobalCallbacks() const {
    return !callbacks_.empty();
  }

  inline bool hasThreadLocalCallbacks() const {
    return !tls_callbacks_.empty();
  }

  inline bool needsInputs() const {
    return has_callbacks_with_inputs_;
  }

  void runStartCallbacks(RecordFunction& rf) {
    rf._setGlobalCallbacksVersion(global_callbacks_version_);
    rf._activeGlobalCallbacks().clear();
    for (size_t cb_idx = 0; cb_idx < callbacks_.size(); ++cb_idx) {
      if (shouldRunCallback(callbacks_[cb_idx], rf.scope())) {
        auto ret = tryRunCallback(callbacks_[cb_idx].start(), rf);
        rf._activeGlobalCallbacks().push_back(ret);
      } else {
        rf._activeGlobalCallbacks().push_back(false);
      }
    }
    for (auto& cb : tls_callbacks_) {
      if (cb.checkScope(rf.scope())) {
        tryRunCallback(cb.start(), rf);
      }
    }
  }

  void runEndCallbacks(RecordFunction& rf) {
    if (rf._globalCallbacksVersion() == global_callbacks_version_) {
      TORCH_CHECK(rf._activeGlobalCallbacks().size() == callbacks_.size());
      for (size_t cb_idx = 0; cb_idx < rf._activeGlobalCallbacks().size(); ++cb_idx) {
        if (!rf._activeGlobalCallbacks()[cb_idx]) {
          continue;
        }
        tryRunCallback(callbacks_[cb_idx].end(), rf);
      }
    } else {
      VLOG(1) << "Global callbacks changed while running a record function, "
              << "you might be calling pushCallback during model "
              << "execution";
    }
    for (auto& cb : tls_callbacks_) {
      if (cb.checkScope(rf.scope())) {
        tryRunCallback(cb.end(), rf);
      }
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
  bool tryRunCallback(std::function<void(const RecordFunction&)>& fn, RecordFunction& rf) {
    try {
      fn(rf);
      return true;
    } catch (const std::exception &e) {
      LOG(WARNING) << "Exception in RecordFunction observer: "
                    << e.what();
      return false;
    } catch (...) {
      LOG(WARNING) << "Exception in RecordFunction observer: unknown";
      return false;
    }
  }

  void recomputeFlags() {
    has_callbacks_with_inputs_ = false;
    for (const auto& cb : callbacks_) {
      has_callbacks_with_inputs_ |= cb.needsInputs();
    }
    for (const auto& cb : tls_callbacks_) {
      has_callbacks_with_inputs_ |= cb.needsInputs();
    }
  }

  inline double samplingProbability(RecordFunctionCallback& cb) const {
    if (cb.isSampled()) {
      return use_global_prob_ ? global_prob_ : cb.samplingProb();
    } else {
      return 1.0;
    }
  }

  inline bool shouldRunCallback(
      RecordFunctionCallback& cb, RecordScope scope) const {
    return cb.checkScope(scope) &&
           (!cb.isSampled() ||
            (sample_zero_one() < samplingProbability(cb)));
  }

  std::vector<RecordFunctionCallback> callbacks_;

  double global_prob_ = 0.0;
  bool use_global_prob_ = false;
  bool has_callbacks_with_inputs_ = false;

  // tracks the current 'version' of global callbacks;
  // every time we push or pop global callbacks, we bump this counter
  uint64_t global_callbacks_version_ = 0;

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
  return manager().hasGlobalCallbacks() || manager().hasThreadLocalCallbacks();
}

bool hasGlobalCallbacks() {
  return manager().hasGlobalCallbacks();
}

bool hasThreadLocalCallbacks() {
  return manager().hasThreadLocalCallbacks();
}

void pushCallback(
    std::function<bool(const RecordFunction&)> start,
    std::function<void(const RecordFunction&)> end,
    bool needs_inputs,
    double sampling_prob,
    std::unordered_set<RecordScope, std::hash<RecordScope>> scopes,
    bool is_thread_local) {
  manager().pushCallback(
      RecordFunctionCallback(std::move(start), std::move(end))
      .needsInputs(needs_inputs)
      .samplingProb(sampling_prob)
      .scopes(std::move(scopes)), is_thread_local);
}

void popCallback(bool is_thread_local) {
  manager().popCallback(is_thread_local);
}

void _runBeforeCallbacks(RecordFunction* rf, const std::string& funcName) {
  TORCH_INTERNAL_ASSERT(rf != nullptr);
  rf->_before(funcName);
}

RecordFunction::RecordFunction(RecordScope scope) : scope_(scope) {
  if (hasCallbacks() && at::_tls_is_record_function_enabled()) {
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
