#pragma once

#include <c10/macros/Macros.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

namespace c10 {

// A RAII, thread local (!) guard that enables or disables inference mode upon
// construction, and sets it back to the original value upon destruction.
struct TORCH_API InferenceMode {
  InferenceMode(bool enabled=true): prev_keyset(c10::impl::tls_local_dispatch_key_set()) {
    // Note [Expected TLS state in InferenceMode]:
    //   InferenceMode: InplaceOrView not in included, Autograd in excluded
    //   NormalMode: InplaceOrView in included, Autograd not in excluded
    //
    // Invariant:
    // - InplaceOrView is never in the excluded set
    // - Autograd is never in the included set
    //
    // 1. Why not put InplaceOrView in the excluded set inside InferenceMode?
    //
    //    For example:
    //    torch::Tensor a = torch::ones({1, 2, 3}).set_requires_grad(true);
    //    torch::Tensor k = a + 2;
    //    {
    //      c10::InferenceMode guard(true);
    //      k.add_(2);
    //    }
    //    `k.add_(2)` still need to go through InplaceOrView kernel so that it's
    //    prepared for future autograd.
    //  2. Why do we need InplaceOrView in included set outside InferenceMode?
    //
    //     Inplace update to inference tensor outside InferenceMode is not allowed.
    //     See Note [Inplace update inference tensor] for more details.
    //     Without going through InplaceOrView kernel, we cannot throw error
    //     for `inference_tensor.add_(1)` case.
    DispatchKeySet included = enabled ? prev_keyset.included_.remove(c10::DispatchKey::InplaceOrView)
         : prev_keyset.included_.add(c10::DispatchKey::InplaceOrView);
    DispatchKeySet excluded = enabled ? (prev_keyset.excluded_ | c10::autograd_dispatch_keyset)
         : (prev_keyset.excluded_ - c10::autograd_dispatch_keyset);
    c10::impl::PODLocalDispatchKeySet cur_keyset {included.raw_repr(), excluded.raw_repr()};
    c10::impl::_force_tls_local_dispatch_key_set(cur_keyset);
  }
  ~InferenceMode() {
    c10::impl::_force_tls_local_dispatch_key_set(prev_keyset);
  }

  static bool is_enabled();

  private:
    c10::impl::LocalDispatchKeySet prev_keyset;
};
} // namespace c10

