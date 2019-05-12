#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/autograd/function.h>

namespace torch { namespace autograd { namespace profiler {

namespace {
std::vector<RecordFunctionCallback> start_callbacks;
std::vector<RecordFunctionCallback> end_callbacks;
size_t callback_needs_inputs = 0;
thread_local RecordFunction* thread_local_func_ = nullptr;

bool is_sampled_callbacks = false;
double sampling_prob = 1.0;
}

void pushCallback(
    RecordFunctionCallback start,
    RecordFunctionCallback end,
    bool needs_inputs) {
  start_callbacks.push_back(start);
  end_callbacks.push_back(end);
  if (callback_needs_inputs > 0 || needs_inputs) {
    ++callback_needs_inputs;
  }
}

void popCallback() {
  if (start_callbacks.empty()) {
    throw std::runtime_error("Empty callbacks stack");
  }
  start_callbacks.pop_back();
  end_callbacks.pop_back();
  if (callback_needs_inputs > 0) {
    --callback_needs_inputs;
  }
}

bool hasCallbacks() {
  return !start_callbacks.empty();
}

bool needsInputs() {
  return callback_needs_inputs > 0;
}

bool isSampledCallbacks() {
  return is_sampled_callbacks;
}

void setSampledCallbacks(bool is_sampled) {
  is_sampled_callbacks = is_sampled;
}

double getSamplingProbability() {
  return sampling_prob;
}

void setSamplingProbability(double prob) {
  sampling_prob = prob;
}

void RecordFunction::before(const char* name, int64_t sequence_nr) {
  if (!hasCallbacks()) {
    return;
  }
  AT_ASSERT(!initialized_);
  name_ = StringView(name);
  sequence_nr_ = sequence_nr;

  initialized_ = true;
  processCallbacks();
}

void RecordFunction::before(std::string name, int64_t sequence_nr) {
  if (!hasCallbacks()) {
    return;
  }
  AT_ASSERT(!initialized_);
  name_ = StringView(std::move(name));
  sequence_nr_ = sequence_nr;

  initialized_ = true;
  processCallbacks();
}

void RecordFunction::before(Function* fn, int64_t sequence_nr) {
  if (!hasCallbacks()) {
    return;
  }
  AT_ASSERT(!initialized_);
  fn_ = fn;
  name_ = StringView(fn->name());
  sequence_nr_ = (sequence_nr >= 0) ? sequence_nr : fn->sequence_nr();

  initialized_ = true;
  processCallbacks();
}

void RecordFunction::processCallbacks() {
  parent_ = thread_local_func_;
  thread_local_func_ = this;

  for (const auto& cb : start_callbacks) {
    cb(*this);
  }
}

RecordFunction::~RecordFunction() {
  if (initialized_) {
    for (const auto& cb : end_callbacks) {
      cb(*this);
    }
    thread_local_func_ = parent_;
  }
}

}}}
