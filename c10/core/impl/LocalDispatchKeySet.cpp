#include <c10/core/impl/LocalDispatchKeySet.h>

#include <iostream>

namespace c10 {
namespace impl {

void _force_tls_local_dispatch_key_set(LocalDispatchKeySet key_set) {
  LocalDispatchKeySetWrapper current_keyset{_get_thread_local_state()};
  current_keyset.set_included(key_set.included_);
  current_keyset.set_excluded(key_set.excluded_);
}

// An RAII guard could snapshot and restore the entire state (entire
// DispatchKeySet) as opposed to only snapshotting and restoring the state of
// its assigned DispatchKeySet. I'm not sure which is better.  If only the RAII
// API is used, the two choices are not distinguishable.
//
// However, if the guard chooses to snapshot and restore the entire
// DispatchKeySet, the interaction with the non-RAII API changes.  Consider this
// sequence of events:
// - An RAII guard is declared for a particular DispatchKeySet, but snapshots
// the entire
//   current DispatchKeySet.
// - A call to the non-RAII API changes the state for DispatchKeys outside the
// assigned
//   set.
// - The RAII guard goes out of scope, restoring the entire DispatchKeySet it
// snapshotted
//   (which restores the state for its own assigned DispatchKey and wipes out
//   the state for the other DispatchKeys set by the non-RAII API).

// RAII API

IncludeDispatchKeyGuard::IncludeDispatchKeyGuard(DispatchKeySet include)
    : tls_wrapper_(_get_thread_local_state()),
      include_(include - tls_wrapper_.included()) {
  if (!include_.empty()) {
    tls_wrapper_.set_included(tls_wrapper_.included() | include_);
  }
}

IncludeDispatchKeyGuard::~IncludeDispatchKeyGuard() {
  if (!include_.empty()) {
    tls_wrapper_.set_included(tls_wrapper_.included() - include_);
  }
}


// Non-RAII API
// Please prefer using the RAII API. See declarations in LocalDispatchKeySet.h
// for details.

bool tls_is_dispatch_key_excluded(DispatchKey x) {
  return LocalDispatchKeySetWrapper(_get_thread_local_state())
      .excluded()
      .has(x);
}

void tls_set_dispatch_key_excluded(DispatchKey x, bool desired_state) {
  LocalDispatchKeySetWrapper tls_wrapper{_get_thread_local_state()};
  bool current_state = tls_wrapper.excluded().has(x);
  if (desired_state != current_state) {
    if (desired_state) {
      tls_wrapper.set_excluded(tls_wrapper.excluded().add(x));
    } else {
      tls_wrapper.set_excluded(tls_wrapper.excluded().remove(x));
    }
  }
}

bool tls_is_dispatch_key_included(DispatchKey x) {
  return LocalDispatchKeySetWrapper(_get_thread_local_state())
      .included()
      .has(x);
}

void tls_set_dispatch_key_included(DispatchKey x, bool desired_state) {
  LocalDispatchKeySetWrapper tls_wrapper{_get_thread_local_state()};
  bool current_state = tls_wrapper.included().has(x);
  if (desired_state != current_state) {
    if (desired_state) {
      tls_wrapper.set_included(tls_wrapper.included().add(x));
    } else {
      tls_wrapper.set_included(tls_wrapper.included().remove(x));
    }
  }
}

bool tls_is_dispatch_keyset_excluded(DispatchKeySet ks) {
  return LocalDispatchKeySetWrapper(_get_thread_local_state())
      .excluded()
      .isSupersetOf(ks);
}

bool tls_is_dispatch_keyset_included(DispatchKeySet ks) {
  return LocalDispatchKeySetWrapper(_get_thread_local_state())
      .included()
      .isSupersetOf(ks);
}
} // namespace impl
} // namespace c10
