#include <c10/core/impl/LocalDispatchKeySet.h>

#include <iostream>

namespace c10 {
namespace impl {

// NB: POD, zero initialized!
thread_local PODLocalDispatchKeySet raw_local_dispatch_key_set;

#if defined(_MSC_VER) || defined(C10_ANDROID)
LocalDispatchKeySet tls_local_dispatch_key_set() {
  return raw_local_dispatch_key_set;
}
#endif // defined(_MSC_VER) || defined(C10_ANDROID)

void _force_tls_local_dispatch_key_set(LocalDispatchKeySet key_set) {
  raw_local_dispatch_key_set = PODLocalDispatchKeySet {
    key_set.included_.raw_repr(),
    key_set.excluded_.raw_repr()
  };
}

// An RAII guard could snapshot and restore the entire state (entire DispatchKeySet) as
// opposed to only snapshotting and restoring the state of its assigned DispatchKeySet.
// I'm not sure which is better.  If only the RAII API is used, the two choices are
// not distinguishable.
//
// However, if the guard chooses to snapshot and restore the entire DispatchKeySet,
// the interaction with the non-RAII API changes.  Consider this sequence of events:
// - An RAII guard is declared for a particular DispatchKeySet, but snapshots the entire
//   current DispatchKeySet.
// - A call to the non-RAII API changes the state for DispatchKeys outside the assigned
//   set.
// - The RAII guard goes out of scope, restoring the entire DispatchKeySet it snapshotted
//   (which restores the state for its own assigned DispatchKey and wipes out the state
//   for the other DispatchKeys set by the non-RAII API).

// RAII API

IncludeDispatchKeyGuard::IncludeDispatchKeyGuard(DispatchKeySet include)
  : tls_(&raw_local_dispatch_key_set)
  , include_(include - tls_->included()) {
  if (!include_.empty()) {
    tls_->set_included(tls_->included() | include_);
  }
}

IncludeDispatchKeyGuard::~IncludeDispatchKeyGuard() {
  if (!include_.empty()) {
    tls_->set_included(tls_->included() - include_);
  }
}

ExcludeDispatchKeyGuard::ExcludeDispatchKeyGuard(DispatchKeySet exclude)
  : tls_(&raw_local_dispatch_key_set)
  , exclude_(exclude - tls_->excluded()) {
  if (!exclude_.empty()) {
    tls_->set_excluded(tls_->excluded() | exclude_);
  }
}

ExcludeDispatchKeyGuard::~ExcludeDispatchKeyGuard() {
  if (!exclude_.empty()) {
    tls_->set_excluded(tls_->excluded() - exclude_);
  }
}

// Non-RAII API
// Please prefer using the RAII API. See declarations in LocalDispatchKeySet.h for details.

bool tls_is_dispatch_key_excluded(DispatchKey x) {
  return raw_local_dispatch_key_set.excluded().has(x);
}

void tls_set_dispatch_key_excluded(DispatchKey x, bool desired_state) {
  auto* tls = &raw_local_dispatch_key_set;
  bool current_state = tls->excluded().has(x);
  if (desired_state != current_state) {
    if (desired_state) {
      tls->set_excluded(tls->excluded().add(x));
    } else {
      tls->set_excluded(tls->excluded().remove(x));
    }
  }
}

bool tls_is_dispatch_key_included(DispatchKey x) {
  return raw_local_dispatch_key_set.included().has(x);
}

void tls_set_dispatch_key_included(DispatchKey x, bool desired_state) {
  auto* tls = &raw_local_dispatch_key_set;
  bool current_state = tls->included().has(x);
  if (desired_state != current_state) {
    if (desired_state) {
      tls->set_included(tls->included().add(x));
    } else {
      tls->set_included(tls->included().remove(x));
    }
  }
}

}} // namespace c10::impl
