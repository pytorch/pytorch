#include <ATen/ThreadLocalState.h>

#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE) && !defined(BUILD_LITE_INTERPRETER)
#include <ATen/autocast_mode.h>
#include <ATen/core/grad_mode.h>
#endif

#include <ATen/record_function.h>
#include <ATen/SavedTensorHooks.h>
#include <ATen/FunctionalTensorWrapper.h>

namespace at {

ThreadLocalState::ThreadLocalState()
    : dispatch_key_(c10::impl::tls_local_dispatch_key_set()),
      debug_info_(c10::ThreadLocalDebugInfo::current()),
      rf_tls_(at::get_record_function_tls_()), functorch_tls_(functorch::getCopyOfFuncTorchTLS()),
      autograd_tls_(c10::AutogradState::get_tls_state()),
      torch_dispatch_mode_state_(c10::impl::TorchDispatchModeTLS::get_state()), python_dispatcher_state_(c10::impl::PythonDispatcherTLS::get_state()),
      python_torch_function_state_(at::impl::PythonTorchFunctionTLS::get_state()),
      saved_tensors_default_hooks_state_(at::SavedTensorDefaultHooks::get_tls_state()), functionalization_reapply_views_state_(at::functionalization::impl::getFunctionalizationReapplyViewsTLS()),
      saved_objects_(at::impl::ThreadLocalPythonObjects::get_state()) {
#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE) && !defined(BUILD_LITE_INTERPRETER)
  for(size_t i=0; i<autocast_dtypes_.size(); i++) {
     autocast_dtypes_[i] = at::autocast::get_autocast_dtype(static_cast<at::DeviceType>(i));
  }
#endif
}

void ThreadLocalState::set_grad_mode(bool enabled) {
  autograd_tls_.set_grad_mode(enabled);
}

void ThreadLocalState::set_multithreading_enabled(bool enabled) {
  autograd_tls_.set_multithreading_enabled(enabled);
}

/* static */
void ThreadLocalState::setThreadLocalState(
    const ThreadLocalState& state) {
  // Note that setting the InferenceMode TLS in this function is ONLY ok because we always
  // restore the dispatch key set TLS at the same time.
  c10::AutogradState::set_tls_state(state.autograd_tls_);

  c10::impl::TorchDispatchModeTLS::set_state(state.torch_dispatch_mode_state_);

  at::impl::PythonTorchFunctionTLS::set_state(state.python_torch_function_state_);

  at::set_record_function_tls_(state.rf_tls_);

  at::SavedTensorDefaultHooks::set_tls_state(state.saved_tensors_default_hooks_state_);

  c10::impl::PythonDispatcherTLS::set_state(state.python_dispatcher_state_);

  c10::ThreadLocalDebugInfo::_forceCurrentDebugInfo(state.debug_info_);

  c10::impl::_force_tls_local_dispatch_key_set(state.dispatch_key_);

  functorch::setFuncTorchTLS(state.functorch_tls_);

  at::functionalization::impl::setFunctionalizationReapplyViewsTLS(state.functionalization_reapply_views_state_);

  at::impl::ThreadLocalPythonObjects::set_state(state.saved_objects_);
#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE) && !defined(BUILD_LITE_INTERPRETER)
  for(size_t i=0; i<state.autocast_dtypes_.size(); i++) {
     at::autocast::set_autocast_dtype(static_cast<at::DeviceType>(i), state.autocast_dtypes_[i]);
  }
#endif
}

} // namespace at
