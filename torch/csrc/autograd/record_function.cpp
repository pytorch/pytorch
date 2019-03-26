#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/autograd/function.h>

namespace torch { namespace autograd { namespace profiler {

namespace {
std::atomic<bool> has_callbacks(false);
std::vector<RecordFunctionCallback> start_callbacks;
std::vector<RecordFunctionCallback> end_callbacks;
thread_local std::shared_ptr<FunctionCallContext> thread_local_ctx;
}

void pushCallback(RecordFunctionCallback start, RecordFunctionCallback end) {
  start_callbacks.push_back(start);
  end_callbacks.push_back(end);
  has_callbacks = true;
}

void pushCallback(RecordFunctionCallback start) {
  pushCallback(start, [](const FunctionCallContext&){});
}

const std::shared_ptr<FunctionCallContext>& currentFunctionCallContext() {
  return thread_local_ctx;
}

void setCurrentFunctionCallContext(
    const std::shared_ptr<FunctionCallContext>& ctx) {
  thread_local_ctx = ctx;
}

void popCallback() {
  if (start_callbacks.empty()) {
    throw std::runtime_error("Empty callbacks stack");
  }
  start_callbacks.pop_back();
  end_callbacks.pop_back();
  has_callbacks = !start_callbacks.empty();
}

// typeid(*fn).name() would avoid an additional string allocation.
// However, typeid(*fn).name() would cause nvtx annotations for all user-defined
// (Python-side) custom autograd function backward() methods to have the same name,
// because they route through the same C++ side class.
// fn->name() ensures that nvtx annotations for custom function backward() methods
// receive a relevant, demangled name.
FunctionCallContext::FunctionCallContext(Function* fn, GetPackedInputsCallback cb)
    : fn_(fn), owned_name_(new std::string(fn->name())), name_ptr_(owned_name_->c_str()),
      sequence_nr_(fn->sequence_nr()), inputs_cb_(cb) {}

FunctionCallContext::FunctionCallContext(
    std::string name, int64_t sequence_nr, GetPackedInputsCallback cb)
    : owned_name_(new std::string(std::move(name))), name_ptr_(owned_name_->c_str()),
      sequence_nr_(sequence_nr), inputs_cb_(cb) {}

FunctionCallContext::FunctionCallContext(
    const char* name_ptr, int64_t sequence_nr, GetPackedInputsCallback cb)
    : name_ptr_(name_ptr), sequence_nr_(sequence_nr), inputs_cb_(cb) {}


RecordFunction::RecordFunction(Function* fn, GetPackedInputsCallback cb) {
  processCallbacks(fn, cb);
}

RecordFunction::RecordFunction(
    std::string name, int64_t sequence_nr, GetPackedInputsCallback cb) {
  processCallbacks(name, sequence_nr, cb);
}

RecordFunction::RecordFunction(
    const char* name, int64_t sequence_nr, GetPackedInputsCallback cb) {
  processCallbacks(name, sequence_nr, cb);
}

template<typename... Args>
void RecordFunction::processCallbacks(Args&&... args) {
  if (!has_callbacks) {
    return;
  }
  ctx_ = std::make_shared<FunctionCallContext>(std::forward<Args>(args)...);
  ctx_->setParent(thread_local_ctx);
  thread_local_ctx = ctx_;

  for (size_t idx = 0; idx < start_callbacks.size(); ++idx) {
    start_callbacks[idx](*ctx_);
  }
}

RecordFunction::~RecordFunction() {
  if (ctx_) {
    for (size_t idx = 0; idx < end_callbacks.size(); ++idx) {
      end_callbacks[idx](*ctx_);
    }
    thread_local_ctx = ctx_->parent();
  }
}

}}}
