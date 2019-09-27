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
thread_local uint64_t raw_included;

#else // defined(CAFFE2_FB_LIMITED_MOBILE_CAPABILITY)

uint64_t raw_excluded = 0;
uint64_t raw_included = 0;

#endif

void set_enabled_via_excluded(TensorTypeId tid, bool enabled) {
  if (enabled) {
    raw_excluded = tls_excluded_tensor_type_set().remove(tid).raw_repr();
  } else {
    raw_excluded = tls_excluded_tensor_type_set().add(tid).raw_repr();
  }
}

void set_enabled_via_included(TensorTypeId tid, bool enabled) {
  if (enabled) {
    raw_included = tls_included_tensor_type_set().add(tid).raw_repr();
  } else {
    raw_included = tls_included_tensor_type_set().remove(tid).raw_repr();
  }
}

}

TensorTypeSet tls_excluded_tensor_type_set() {
  return TensorTypeSet(TensorTypeSet::RAW, raw_excluded);
}

TensorTypeSet tls_included_tensor_type_set() {
  return TensorTypeSet(TensorTypeSet::RAW, raw_included);
}

bool tls_variable_is_enabled() {
  return !tls_excluded_tensor_type_set().has(TensorTypeId::VariableTensorId);
}

void tls_variable_set_enabled(bool enabled) {
  set_enabled_via_excluded(TensorTypeId::VariableTensorId, enabled);
}

bool TESTING_ONLY_tls_generic_mode_is_enabled() {
  return tls_included_tensor_type_set().has(TensorTypeId::TESTING_ONLY_GenericModeTensorId);
}

void TESTING_ONLY_tls_generic_mode_set_enabled(bool enabled) {
  set_enabled_via_included(TensorTypeId::TESTING_ONLY_GenericModeTensorId, enabled);
}

}} // namespace c10::impl
