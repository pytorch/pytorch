#pragma once

namespace at {
namespace detail {

// Suspends all backends in the dispatcher and diverts op requests to the meta
// dispatch key.
void suspendBackendDispatch();

// Restores the dispatcher to its original state.
void restoreBackendDispatch();

bool backendDispatchSuspended() noexcept;

} // namespace detail
} // namespace at
