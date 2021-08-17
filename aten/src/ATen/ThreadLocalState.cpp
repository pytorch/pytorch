#include <ATen/ThreadLocalState.h>

#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
#include <ATen/core/grad_mode.h>
#endif

#include <ATen/record_function.h>
#include <ATen/SavedTensorHooks.h>

namespace at {

ThreadLocalState::ThreadLocalState(bool keep_grad_mode)
    : dispatch_key_(c10::impl::tls_local_dispatch_key_set()),
      debug_info_(c10::ThreadLocalDebugInfo::current()),
      autograd_tls_(c10::AutogradState::get_tls_state()) {
  rf_tls_ = at::get_record_function_tls_();
  saved_tensors_default_hooks_ = SavedTensorDefaultHooks::get_hooks();

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
    c10::AutogradState::set_tls_state(state.autograd_tls_);
  } else {
    auto new_state = c10::AutogradState(/* grad_mode */ c10::AutogradState::get_tls_state().get_grad_mode(),
                                        /* inference_mode */ state.autograd_tls_.get_inference_mode());
    c10::AutogradState::set_tls_state(new_state);
  }
#else
  c10::AutogradState::set_tls_state(state.autograd_tls_);
#endif

  at::set_record_function_tls_(state.rf_tls_);

  SavedTensorDefaultHooks::set_hooks(
      state.saved_tensors_default_hooks_.first,
      state.saved_tensors_default_hooks_.second);

  c10::ThreadLocalDebugInfo::_forceCurrentDebugInfo(state.debug_info_);

  c10::impl::_force_tls_local_dispatch_key_set(state.dispatch_key_);
}

} // namespace at
