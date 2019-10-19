#include <ATen/core/ATenDispatch.h>

namespace at {

ATenDispatch & globalATenDispatch() {
  static ATenDispatch singleton;
  return singleton;
}

void ATenOpTable::reportError(TensorTypeId tid) const {
  std::ostringstream oss;
  bool first = true;
  for (int64_t i = 0; i < static_cast<int64_t>(TensorTypeId::NumTensorIds); i++) {
    if (function_table_[i] != nullptr) {
      if (!first) oss << ", ";
      oss << toString(static_cast<TensorTypeId>(i));
      first = false;
    }
  }

  // If there is no fallback dispatch, and dispatch failed because we didn't
  // find any valid keys to dispatch on, this usually means the user gave
  // us a non-empty list of tensors.  So report a better error in this case.
  // TODO: Maybe we should reword this error message
  if (tid == TensorTypeId::UndefinedTensorId) {
    TORCH_CHECK(false,
          "There were no tensor arguments to this function (e.g., you passed an "
          "empty list of Tensors), but no fallback function is registered for schema ", schema_,
          ".  This usually means that this function requires a non-empty list of Tensors.  "
          "Available functions are ", oss.str())
  }
  TORCH_CHECK(false,
    "No function is registered for schema ", schema_, " on tensor type ", toString(tid),
    "; available functions are ", oss.str());
}

} // namespace at
