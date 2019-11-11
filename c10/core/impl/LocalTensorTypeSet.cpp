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

// We could have also just snapshotted the entire state.  I'm not sure which is
// better; but right now only the guard API is allowed so the two cases are
// not distinguishable.

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

TensorTypeSet tls_get_local_included_tensor_type_set() {
  return raw_local_tensor_type_set.included();
}
void tls_set_local_included_tensor_type_set(TensorTypeSet ts) {
  raw_local_tensor_type_set.set_included(ts);
}
TensorTypeSet tls_get_local_excluded_tensor_type_set() {
  return raw_local_tensor_type_set.excluded();
}
void tls_set_local_excluded_tensor_type_set(TensorTypeSet ts) {
  raw_local_tensor_type_set.set_excluded(ts);
}

}} // namespace c10::impl
