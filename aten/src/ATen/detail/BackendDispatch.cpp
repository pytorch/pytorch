#include <ATen/detail/BackendDispatch.h>

#include <cstdint>

#include <c10/core/DispatchKey.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

using c10::impl::tls_set_dispatch_key_excluded;
using c10::impl::tls_set_dispatch_key_included;

namespace at {
namespace detail {
namespace {

void excludeBackendDispatchKeys(bool desired_state) {
  // Add all backend keys except meta to the TLS exclude list. This will force
  // the dispatcher to suspend all tensor operations.
  auto end = static_cast<std::uint8_t>(DispatchKey::EndOfBackendKeys);
  for (std::uint8_t i = 1; i <= end; i++) {
    auto dk = static_cast<DispatchKey>(i);
    if (dk == DispatchKey::Meta) {
      continue;
    }
    tls_set_dispatch_key_excluded(dk, desired_state);
  }

  // In addition to the backend ops, we also have to stop the backend selection
  // logic; otherwise, some operations such as `empty` can escape the exclusion
  // list.
  tls_set_dispatch_key_excluded(DispatchKey::BackendSelect, desired_state);
}

} // namespace

void suspendBackendDispatch() {
  excludeBackendDispatchKeys(/*desired_state*/true);

  // Now that we have excluded all backends from the dispatcher, enable meta
  // dispatch key which effectively diverts all backend ops to meta.
  tls_set_dispatch_key_included(DispatchKey::Meta, /*desired_state*/true);
}

void restoreBackendDispatch() {
  excludeBackendDispatchKeys(/*desired_state*/false);

  tls_set_dispatch_key_included(DispatchKey::Meta, /*desired_state*/false);
}

bool backendDispatchSuspended() noexcept {
  return c10::impl::tls_is_dispatch_key_excluded(DispatchKey::BackendSelect);
}

} // namespace detail
} // namespace at
