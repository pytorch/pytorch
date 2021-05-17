#pragma once

#include <c10/core/GradMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/macros/Macros.h>

namespace c10 {

// A RAII, thread local (!) guard that enables or disables inference mode upon
// construction, and sets it back to the original value upon destruction.
struct TORCH_API InferenceMode {
  // Note [Expected TLS state in InferenceMode]:
  //   InferenceMode: ADInplaceOrView not in
  //   raw_local_dispatch_key_set.included(),
  //                  Autograd in raw_local_dispatch_key_set.excluded()
  //                  GradMode is disabled.
  //   NormalMode: ADInplaceOrView in raw_local_dispatch_key_set.included(),
  //               Autograd not in raw_local_dispatch_key_set.excluded()
  //               GradMode is enabled by default unless toggled manually
  //               through other APIs, e.g. NoGradGuard.
  //
  // Invariant:
  // - ADInplaceOrView is never in the excluded set
  // - Autograd is never in the included set
  // - Setting InferenceMode will set GradMode accordingly, but not vice versa.
  //
  //  1. Why do we put ADInplaceOrView in included set outside InferenceMode?
  //
  //     Inplace update to inference tensor outside InferenceMode is not
  //     allowed. See Note [Inplace update inference tensor] for more details.
  //     Without going through ADInplaceOrView kernel, we cannot throw error
  //     for `inference_tensor.add_(1)` case.
  //
  // 2. Why not put ADInplaceOrView in the excluded set inside InferenceMode?
  //
  //    For example:
  //    torch::Tensor a = torch::ones({1, 2, 3}).set_requires_grad(true);
  //    torch::Tensor k = a + 2;
  //    {
  //      c10::InferenceMode guard(true);
  //      k.add_(2);
  //    }
  //    `k.add_(2)` still need to go through ADInplaceOrView kernel so that it's
  //    prepared for future autograd.
  //
  // 3. Why does setting InferenceMode also set GradMode?
  //
  //    This is required since InferenceMode is a faster and more restricive
  //    version of NoGradGuard. All runtime checks using GradMode::is_enabled()
  //    are applicable to InferenceMode as well, e.g.
  //    `tensorTypeInCurrentExecutionContext` in interpreter.cpp.
  InferenceMode(bool enabled = true)
      : prev_mode(InferenceMode::is_enabled()),
        prev_keyset(c10::impl::tls_local_dispatch_key_set()),
        grad_mode(at::AutoGradMode(!enabled)) {
    _set_enabled(enabled);
    DispatchKeySet included = enabled
        ? prev_keyset.included_.remove(c10::DispatchKey::ADInplaceOrView)
        : prev_keyset.included_.add(c10::DispatchKey::ADInplaceOrView);
    DispatchKeySet excluded = enabled
        ? (prev_keyset.excluded_ | c10::autograd_dispatch_keyset)
        : (prev_keyset.excluded_ - c10::autograd_dispatch_keyset);
    c10::impl::PODLocalDispatchKeySet cur_keyset;
    cur_keyset.set_included(included);
    cur_keyset.set_excluded(excluded);
    c10::impl::_force_tls_local_dispatch_key_set(cur_keyset);
  }

  ~InferenceMode() {
    _set_enabled(prev_mode);
    c10::impl::_force_tls_local_dispatch_key_set(prev_keyset);
  }
  static bool is_enabled();
  // _set_enabled() is not user facing and should be only used in
  // ThreadLocalState.cpp.
  static void _set_enabled(bool enabled);

 private:
  bool prev_mode;
  c10::impl::LocalDispatchKeySet prev_keyset;
  at::AutoGradMode grad_mode;
};
} // namespace c10
