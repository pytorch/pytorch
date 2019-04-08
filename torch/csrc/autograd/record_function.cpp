#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/autograd/function.h>

namespace torch { namespace autograd { namespace profiler {

namespace {
bool has_callbacks = false;
std::vector<RecordFunctionCallback> start_callbacks;
std::vector<RecordFunctionCallback> end_callbacks;
thread_local RecordFunction* thread_local_func_ = nullptr;
}

void pushCallback(RecordFunctionCallback start, RecordFunctionCallback end) {
  start_callbacks.push_back(start);
  end_callbacks.push_back(end);
  has_callbacks = true;
}

void pushCallback(RecordFunctionCallback start) {
  pushCallback(start, [](const RecordFunction&){});
}

void popCallback() {
  if (start_callbacks.empty()) {
    throw std::runtime_error("Empty callbacks stack");
  }
  start_callbacks.pop_back();
  end_callbacks.pop_back();
  has_callbacks = !start_callbacks.empty();
}

RecordFunction::RecordFunction(Function* fn, GetPackedInputsCallback cb) {
  if (!has_callbacks) {
    return;
  }
  fn_ = fn;
  name_ = StringView(fn->name());
  sequence_nr_ = fn->sequence_nr();
  inputs_cb_ = cb;
  processCallbacks();
}

RecordFunction::RecordFunction(
    std::string name, int64_t sequence_nr, GetPackedInputsCallback cb) {
  if (!has_callbacks) {
    return;
  }
  name_ = StringView(std::move(name));
  sequence_nr_ = sequence_nr;
  inputs_cb_ = cb;
  processCallbacks();
}

RecordFunction::RecordFunction(
    const char* name, int64_t sequence_nr, GetPackedInputsCallback cb) {
  if (!has_callbacks) {
    return;
  }
  name_ = StringView(name);
  sequence_nr_ = sequence_nr;
  inputs_cb_ = cb;
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
  if (has_callbacks) {
    for (const auto& cb : end_callbacks) {
      cb(*this);
    }
    thread_local_func_ = parent_;
  }
}

}}}
