#include <c10/core/impl/LocalTensorTypeSet.h>

namespace c10 {
namespace impl {

namespace {

/// In the CAFFE2_FB_LIMITED_MOBILE_CAPABILITY build setting,
/// thread_local is not supported. In that case, we don't provide
/// `at::NonVariableTypeMode`.
#ifndef CAFFE2_FB_LIMITED_MOBILE_CAPABILITY

thread_local TensorTypeSet valid(TensorTypeSet::FULL);

#else // defined(CAFFE2_FB_LIMITED_MOBILE_CAPABILITY)

TensorTypeSet valid(TensorTypeSet::FULL)

#endif

}

bool tls_variable_is_enabled() {
  return valid.has(TensorTypeId::VariableTensorId);
}

void tls_variable_set_enabled(bool enabled) {
  if (enabled) {
    valid = valid.add(TensorTypeId::VariableTensorId);
  } else {
    valid = valid.remove(TensorTypeId::VariableTensorId);
  }
}

TensorTypeSet tls_valid_tensor_type_set() {
  return valid;
}

}} // namespace c10::impl
