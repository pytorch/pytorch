#include <ATen/PythonModeTLS.h>

namespace at { namespace impl {

#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
thread_local std::shared_ptr<TorchDispatchTypeObject> pythonModeState;

void PythonModeTLS::set_state(const std::shared_ptr<TorchDispatchTypeObject>& state) {
  pythonModeState = state;
  if (state) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::PythonMode, true);
  } else {
    PythonModeTLS::reset_state();
  }
}

const std::shared_ptr<TorchDispatchTypeObject>& PythonModeTLS::get_state() {
  return pythonModeState;
}

void PythonModeTLS::reset_state() {
  pythonModeState.reset((TorchDispatchTypeObject*)nullptr);
  c10::impl::tls_set_dispatch_key_included(DispatchKey::PythonMode, false);
}

static void dispatchToPython(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto* interpreter = pythonModeState->pyinterpreter();
  interpreter->dispatch(op, stack, pythonModeState);
}
#else
void PythonModeTLS::set_state(const std::shared_ptr<TorchDispatchTypeObject>& state) {
  TORCH_INTERNAL_ASSERT(false, "PythonModeTLS not enabled in build");
}

const std::shared_ptr<TorchDispatchTypeObject>& PythonModeTLS::get_state() {
  TORCH_INTERNAL_ASSERT(false, "PythonModeTLS not enabled in build");
}

void PythonModeTLS::reset_state() {
  TORCH_INTERNAL_ASSERT(false, "PythonModeTLS not enabled in build");
}

static void dispatchToPython(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_INTERNAL_ASSERT(false, "PythonModeTLS not enabled in build");
}
#endif

TORCH_LIBRARY_IMPL(_, PythonMode, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&dispatchToPython>());
}


}}
