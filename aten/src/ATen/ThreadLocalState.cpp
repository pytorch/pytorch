#include <ATen/ThreadLocalState.h>

#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
#include <ATen/core/grad_mode.h>
#endif

#include <ATen/record_function.h>

namespace at {

ThreadLocalState::ThreadLocalState(bool keep_grad_mode)
    : dispatch_key_(c10::impl::tls_local_dispatch_key_set()),
      debug_info_(c10::ThreadLocalDebugInfo::current()),
      autograd_tls_(c10::AutogradTLS::get_mode()) {
  rf_tls_ = at::get_record_function_tls_();

#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
  keep_grad_mode_ = keep_grad_mode;
#endif
  bumped_record_all_functions_ = at::checkRecordAllFunctions();
}

/* static */
void ThreadLocalState::setThreadLocalState(
    const ThreadLocalState& state) {
  // Note that setting the InferenceMode TLS in this function is ONLY ok because we always
  // restore the dispatch key set TLS at the same time.
#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
  if (state.keep_grad_mode_) {
    c10::AutogradTLS::set_mode(state.autograd_tls_);
  } else {
    c10::AutogradTLS::set_inference_mode(state.autograd_tls_ & c10::AutogradTLS::INFERENCE_MODE_MASK);
  }
#else
  c10::AutogradTLS::set_mode(state.autograd_tls_);
#endif

  at::set_record_function_tls_(state.rf_tls_);

  c10::ThreadLocalDebugInfo::_forceCurrentDebugInfo(state.debug_info_);

  c10::impl::_force_tls_local_dispatch_key_set(state.dispatch_key_);
}

} // namespace at
