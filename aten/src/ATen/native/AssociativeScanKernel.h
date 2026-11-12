#pragma once

#include <ATen/core/TensorBase.h>
#include <ATen/native/DispatchStub.h>

#include <c10/core/TensorImpl.h>

#include <string>
#include <vector>

namespace at::native {

// Both input and output tensors are contiguous and the scan is always
// performed along the innermost (contiguous) dimension. `combine_mode` is one
// of "add", "mul", "max", "min".
using associative_scan_fn = void (*)(
    const TensorBase& result,
    const TensorBase& self,
    const std::string& combine_mode);

// Multi-tensor overload. Only combine_mode == "linear_recurrence" (arity 2) is
// supported; every tensor in `result`/`self` is contiguous with the scan along
// the innermost dimension.
using associative_scan_tensor_list_fn = void (*)(
    const std::vector<TensorBase>& result,
    const std::vector<TensorBase>& self,
    const std::string& combine_mode);

DECLARE_DISPATCH(associative_scan_fn, associative_scan_stub)
DECLARE_DISPATCH(
    associative_scan_tensor_list_fn,
    associative_scan_tensor_list_stub)

} // namespace at::native
