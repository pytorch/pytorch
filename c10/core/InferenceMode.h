#pragma once

#include <c10/macros/Macros.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

namespace c10 {

// A RAII, thread local (!) guard that enables or disables inference mode upon
// construction, and sets it back to the original value upon destruction.
struct TORCH_API InferenceMode {
  InferenceMode(bool enabled=true) : prev_mode(is_enabled()),
   prev_keyset(c10::impl::tls_local_dispatch_key_set()) {
    // Expected state:
    //   InferenceMode: no InplaceOrView in included, Autograd in excluded
    //   NormalMode: InplaceOrView in included, no Autograd in excluded
    //
    // 1. When InferenceMode is enabled, Autograd dispatch keys are excluded
    //    but not InplaceOrView key.
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
    //  2. When InferenceMode is disabled, InplaceOrView must be added
    //     to included set.
    //
    //     For example:
    //     torch::Tensor a;
    //     {
    //       c10::InferenceMode guard(true);
    //       torch::Tensor in = torch::ones({2, 2});
    //       a = in.view({1, 4});
    //     }
    //     torch::Tensor c = a.view({4, 1}); // (*)
    //     If we don't add InplaceOrView to included set, (*) will skip its as_view
    //     setup entirely which is wrong.
    //     By going through InplaceOrView kernel, we can either throw an error
    //     or setup as_view accordingly.

    DispatchKeySet included = enabled ? prev_keyset.included_.remove(c10::DispatchKey::InplaceOrView)
         : prev_keyset.included_.add(c10::DispatchKey::InplaceOrView);
    DispatchKeySet excluded = enabled ? (prev_keyset.excluded_ | c10::autograd_dispatch_keyset)
         : (prev_keyset.excluded_ - c10::autograd_dispatch_keyset);
    c10::impl::PODLocalDispatchKeySet cur_keyset {included.raw_repr(), excluded.raw_repr()};
    c10::impl::_force_tls_local_dispatch_key_set(cur_keyset);
    set_enabled(enabled);
  }
  ~InferenceMode() {
    set_enabled(prev_mode);
    c10::impl::_force_tls_local_dispatch_key_set(prev_keyset);
  }

  static bool is_enabled();

  private:
    static void set_enabled(bool enabled);

    bool prev_mode;
    c10::impl::LocalDispatchKeySet prev_keyset;
};
} // namespace c10

