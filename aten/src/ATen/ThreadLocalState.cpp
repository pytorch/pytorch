#include <ATen/ThreadLocalState.h>

#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
#include <ATen/core/grad_mode.h>
#endif

#include <ATen/record_function.h>

namespace at {

ThreadLocalState::ThreadLocalState(bool keep_grad_mode)
    : dispatch_key_(c10::impl::tls_local_dispatch_key_set()),
      debug_info_(c10::ThreadLocalDebugInfo::current()),
      observers_enabled_(at::isRecordFunctionEnabled()) {
  callbacks_ = _getTLSCallbacks();

#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
  keep_grad_mode_ = keep_grad_mode;
  if (keep_grad_mode_) {
    grad_mode_enabled_ = GradMode::is_enabled();
  }
#endif
}

/* static */
void ThreadLocalState::setThreadLocalState(
    const ThreadLocalState& state) {
#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
  if (state.keep_grad_mode_) {
    GradMode::set_enabled(state.grad_mode_enabled_);
  }
#endif

  _setTLSCallbacks(state.callbacks_);

  at::enableRecordFunction(state.observers_enabled_);

  c10::ThreadLocalDebugInfo::_forceCurrentDebugInfo(state.debug_info_);

  c10::impl::_force_tls_local_dispatch_key_set(state.dispatch_key_);
}

} // namespace at
