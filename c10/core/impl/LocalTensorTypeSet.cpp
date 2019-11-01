#include <c10/core/impl/LocalTensorTypeSet.h>

#include <iostream>

namespace c10 {
namespace impl {

namespace {

/// In the CAFFE2_FB_LIMITED_MOBILE_CAPABILITY build setting,
/// thread_local is not supported. In that case, we don't provide
/// `at::NonVariableTypeMode`.
#ifndef CAFFE2_FB_LIMITED_MOBILE_CAPABILITY

// NB: Zero initialized!
thread_local uint64_t raw_excluded;

#else // defined(CAFFE2_FB_LIMITED_MOBILE_CAPABILITY)

uint64_t raw_excluded = 0;

#endif

}

TensorTypeSet tls_excluded_tensor_type_set() {
  return TensorTypeSet(TensorTypeSet::RAW, raw_excluded);
}

bool tls_variable_is_enabled() {
  return !tls_excluded_tensor_type_set().has(TensorTypeId::VariableTensorId);
}

void tls_variable_set_enabled(bool enabled) {
  if (enabled) {
    raw_excluded = tls_excluded_tensor_type_set().remove(TensorTypeId::VariableTensorId).raw_repr();
  } else {
    raw_excluded = tls_excluded_tensor_type_set().add(TensorTypeId::VariableTensorId).raw_repr();
  }
}

}} // namespace c10::impl
