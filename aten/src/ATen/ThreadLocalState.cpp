#include <ATen/ThreadLocalState.h>

#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
#include <ATen/core/grad_mode.h>
#endif

#include <ATen/record_function.h>

namespace at {

ThreadLocalState::ThreadLocalState()
    : dispatch_key_(c10::impl::tls_local_dispatch_key_set()),
      debug_info_(c10::ThreadLocalDebugInfo::current()),
      autograd_tls_(c10::AutogradTLS::get_mode()) {
  rf_tls_ = at::get_record_function_tls_();
  bumped_record_all_functions_ = at::checkRecordAllFunctions();
}

void ThreadLocalState::set_grad_mode(bool enabled) {
  if (enabled) {
    autograd_tls_ |= AutogradTLS::GRAD_MODE_MASK;
  } else {
    autograd_tls_ &= ~AutogradTLS::GRAD_MODE_MASK;
  }
}

/* static */
void ThreadLocalState::setThreadLocalState(
    const ThreadLocalState& state) {
  // Note that setting the InferenceMode TLS in this function is ONLY ok because we always
  // restore the dispatch key set TLS at the same time.
  c10::AutogradTLS::set_mode(state.autograd_tls_);

  at::set_record_function_tls_(state.rf_tls_);

  c10::ThreadLocalDebugInfo::_forceCurrentDebugInfo(state.debug_info_);

  c10::impl::_force_tls_local_dispatch_key_set(state.dispatch_key_);
}

} // namespace at
