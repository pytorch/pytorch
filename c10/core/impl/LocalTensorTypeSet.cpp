#include <c10/core/impl/LocalTensorTypeSet.h>

#include <iostream>

namespace c10 {
namespace impl {

namespace {

/// In the CAFFE2_FB_LIMITED_MOBILE_CAPABILITY build setting,
/// thread_local is not supported.
#ifndef CAFFE2_FB_LIMITED_MOBILE_CAPABILITY

// NB: POD, zero initialized!
thread_local PODLocalTensorTypeSet raw_local_tensor_type_set;

#else // defined(CAFFE2_FB_LIMITED_MOBILE_CAPABILITY)

static PODLocalTensorTypeSet raw_local_tensor_type_set;

#endif

} // anonymous namespace

LocalTensorTypeSet tls_local_tensor_type_set() {
  return raw_local_tensor_type_set;
}

// An RAII guard could snapshot and restore the entire state (entire TensorTypeSet) as
// opposed to only snapshotting and restoring the state of its assigned TensorTypeId.
// I'm not sure which is better.  If only the RAII API is used, the two choices are
// not distinguishable.
//
// However, if the guard chooses to snapshot and restore the entire TensorTypeSet,
// the interaction with the non-RAII API changes.  Consider this sequence of events:
// - An RAII guard is declared for a particular TensorTypeId, but snapshots the entire
//   current TensorTypeSet.
// - A call to the non-RAII API changes the state for a different TensorTypeId.
// - The RAII guard goes out of scope, restoring the entire TensorTypeSet it snapshotted
//   (which restores the state for its own assigned TensorTypeId and wipes out the state
//   for the other TensorTypeId set by the non-RAII API).

// RAII API

IncludeTensorTypeIdGuard::IncludeTensorTypeIdGuard(TensorTypeId x)
  : tls_(&raw_local_tensor_type_set)
  , id_(x)
  , prev_state_(tls_->included().has(x)) {
  if (!prev_state_) {
    tls_->set_included(tls_->included().add(x));
  }
}

IncludeTensorTypeIdGuard::~IncludeTensorTypeIdGuard() {
  if (!prev_state_) {
    tls_->set_included(tls_->included().remove(id_));
  }
}

ExcludeTensorTypeIdGuard::ExcludeTensorTypeIdGuard(TensorTypeId x)
  : tls_(&raw_local_tensor_type_set)
  , id_(x)
  , prev_state_(tls_->excluded().has(x)) {
  if (!prev_state_) {
    tls_->set_excluded(tls_->excluded().add(x));
  }
}

ExcludeTensorTypeIdGuard::~ExcludeTensorTypeIdGuard() {
  if (!prev_state_) {
    tls_->set_excluded(tls_->excluded().remove(id_));
  }
}

// Non-RAII API
// Please prefer using the RAII API. See declarations in LocalTensorTypeSet.h for details.

bool tls_is_tensor_type_id_excluded(TensorTypeId x) {
  return raw_local_tensor_type_set.excluded().has(x);
}

void tls_set_tensor_type_id_excluded(TensorTypeId x, bool desired_state) {
  auto* tls = &raw_local_tensor_type_set;
  bool current_state = tls->excluded().has(x);
  if (desired_state != current_state) {
    if (desired_state) {
      tls->set_excluded(tls->excluded().add(x));
    } else {
      tls->set_excluded(tls->excluded().remove(x));
    }
  }
}

bool tls_is_tensor_type_id_included(TensorTypeId x) {
  return raw_local_tensor_type_set.included().has(x);

}

void tls_set_tensor_type_id_included(TensorTypeId x, bool desired_state) {
  auto* tls = &raw_local_tensor_type_set;
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
