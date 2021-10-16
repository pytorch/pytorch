#include <ATen/ThreadLocalState.h>

#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
#include <ATen/core/grad_mode.h>
#endif

#include <ATen/record_function.h>
#include <ATen/SavedTensorHooks.h>

namespace at {

ThreadLocalState::ThreadLocalState()
    : dispatch_key_(c10::impl::tls_local_dispatch_key_set()),
      debug_info_(c10::ThreadLocalDebugInfo::current()),
      autograd_tls_(c10::AutogradState::get_tls_state()) {
  rf_tls_ = at::get_record_function_tls_();
  saved_tensors_default_hooks_ = SavedTensorDefaultHooks::get_hooks();

  bumped_record_all_functions_ = at::checkRecordAllFunctions();
  python_mode_state_ = at::impl::PythonModeTLS::get_state();
}

void ThreadLocalState::set_grad_mode(bool enabled) {
  autograd_tls_.set_grad_mode(enabled);
}

/* static */
void ThreadLocalState::setThreadLocalState(
    const ThreadLocalState& state) {
  // Note that setting the InferenceMode TLS in this function is ONLY ok because we always
  // restore the dispatch key set TLS at the same time.
  c10::AutogradState::set_tls_state(state.autograd_tls_);

  at::impl::PythonModeTLS::set_state(state.python_mode_state_);

  at::set_record_function_tls_(state.rf_tls_);

  SavedTensorDefaultHooks::set_hooks(
      state.saved_tensors_default_hooks_.first,
      state.saved_tensors_default_hooks_.second);

  c10::ThreadLocalDebugInfo::_forceCurrentDebugInfo(state.debug_info_);

  c10::impl::_force_tls_local_dispatch_key_set(state.dispatch_key_);
}

} // namespace at
