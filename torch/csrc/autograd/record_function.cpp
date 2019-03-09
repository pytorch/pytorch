#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/autograd/function.h>

namespace torch { namespace autograd { namespace profiler {

namespace {
std::atomic<bool> has_callbacks(false);
std::vector<CallbackCreator> cbc;
thread_local std::shared_ptr<FunctionCallContext> thread_local_ctx;
}

void pushCallback(CallbackCreator callback_creator) {
  cbc.push_back(callback_creator);
  has_callbacks = true;
}

const std::shared_ptr<FunctionCallContext>& currentFunctionCallContext() {
  return thread_local_ctx;
}

void setCurrentFunctionCallContext(
    const std::shared_ptr<FunctionCallContext>& ctx) {
  thread_local_ctx = ctx;
}

void popCallback() {
  if (cbc.empty()) {
    throw std::runtime_error("Empty callbacks stack");
  }
  cbc.pop_back();
  has_callbacks = !cbc.empty();
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
  for (size_t idx = 0; idx < cbc.size(); ++idx) {
    auto cb_ptr = cbc[idx](*ctx_);
    cbs_.push_back(std::move(cb_ptr));
  }
}

RecordFunction::~RecordFunction() {
  if (ctx_) {
    thread_local_ctx = ctx_->parent();
  }
}

}}}
