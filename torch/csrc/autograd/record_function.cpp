#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/autograd/function.h>

namespace torch { namespace autograd { namespace profiler {

namespace {
RecordFunctionState record_function_state = RecordFunctionState::NO_CALLBACKS;
std::vector<RecordFunctionCallback> start_callbacks;
std::vector<RecordFunctionCallback> end_callbacks;
size_t callback_needs_inputs = 0;
thread_local RecordFunction* thread_local_func_ = nullptr;
}

void pushCallback(
    RecordFunctionCallback start, RecordFunctionCallback end, bool needs_inputs) {
  start_callbacks.push_back(start);
  end_callbacks.push_back(end);
  if (callback_needs_inputs > 0 || needs_inputs) {
    ++callback_needs_inputs;
  }
  if (callback_needs_inputs > 0) {
    record_function_state = RecordFunctionState::NAME_AND_INPUTS;
  } else {
    record_function_state = RecordFunctionState::NAME_ONLY;
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
  if (start_callbacks.empty()) {
    record_function_state = RecordFunctionState::NO_CALLBACKS;
  } else if (callback_needs_inputs > 0) {
    record_function_state = RecordFunctionState::NAME_AND_INPUTS;
  } else {
    record_function_state = RecordFunctionState::NAME_ONLY;
  }
}

TORCH_API RecordFunctionState recordFunctionState() {
  return record_function_state;
}

RecordFunction::RecordFunction(Function* fn) {
  before(fn);
}

RecordFunction::RecordFunction(
    std::string name, int64_t sequence_nr) {
  before(std::move(name), c10::nullopt, sequence_nr);
}

RecordFunction::RecordFunction(
    const char* name, int64_t sequence_nr) {
  before(name, c10::nullopt, sequence_nr);
}

void RecordFunction::before(
    Function* fn, c10::optional<at::ArrayRef<c10::IValue>> inputs) {
  if (record_function_state == RecordFunctionState::NO_CALLBACKS) {
    return;
  }
  if (initialized_) {
    throw std::runtime_error("Double initializing RecordFunction");
  }
  fn_ = fn;
  name_ = StringView(fn->name());
  sequence_nr_ = fn->sequence_nr();
  if (inputs) {
    copyInputs(*inputs);
  }
  initialized_ = true;
  processCallbacks();
}

void RecordFunction::before(
    std::string name, c10::optional<at::ArrayRef<c10::IValue>> inputs, int64_t sequence_nr) {
  if (record_function_state == RecordFunctionState::NO_CALLBACKS) {
    return;
  }
  if (initialized_) {
    throw std::runtime_error("Double initializing RecordFunction");
  }
  name_ = StringView(std::move(name));
  sequence_nr_ = sequence_nr;
  if (inputs) {
    copyInputs(*inputs);
  }
  initialized_ = true;
  processCallbacks();
}

void RecordFunction::before(
    const char* name, c10::optional<at::ArrayRef<c10::IValue>> inputs, int64_t sequence_nr) {
  if (record_function_state == RecordFunctionState::NO_CALLBACKS) {
    return;
  }
  if (initialized_) {
    throw std::runtime_error("Double initializing RecordFunction");
  }
  name_ = StringView(name);
  sequence_nr_ = sequence_nr;
  if (inputs) {
    copyInputs(*inputs);
  }
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
  if (initialized_ &&
      record_function_state != RecordFunctionState::NO_CALLBACKS) {
    for (const auto& cb : end_callbacks) {
      cb(*this);
    }
    thread_local_func_ = parent_;
  }
}

}}}
