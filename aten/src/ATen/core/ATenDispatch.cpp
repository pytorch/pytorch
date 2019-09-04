#include <ATen/core/ATenDispatch.h>

namespace at {

ATenDispatch & globalATenDispatch() {
  static ATenDispatch singleton;
  return singleton;
}

void* ATenOpTable::getFallbackOp(TensorTypeId tid) const {
  // TODO: an alternate strategy here would be to mask out the dead key
  // and then redispatch gain (automatic delegation).  I haven't done this
  // for now to make it easier to smoke out error cases.
  if (function_table_[static_cast<int64_t>(TensorTypeId::UndefinedTensorId)] == nullptr) {
    // If there is no fallback dispatch, and dispatch failed because we didn't
    // find any valid keys to dispatch on, this usually means the user gave
    // us a non-empty list of tensors.  So report a better error in this case.
    // TODO: Maybe we should reword this error message
    if (tid == TensorTypeId::UndefinedTensorId) {
      TORCH_CHECK(false, "expected a non-empty list of Tensors")
    }
    std::ostringstream oss;
    bool first = true;
    for (int64_t i = 0; i < static_cast<int64_t>(TensorTypeId::NumTensorIds); i++) {
      if (function_table_[i] != nullptr) {
        if (!first) oss << ", ";
        oss << toString(static_cast<TensorTypeId>(i));
        first = false;
      }
    }
    TORCH_CHECK(false,
      "No function is registered for schema ", schema_, " on tensor type ", toString(tid),
      "; available functions are ", oss.str());
  }
  return function_table_[static_cast<int64_t>(TensorTypeId::UndefinedTensorId)];
}

} // namespace at
