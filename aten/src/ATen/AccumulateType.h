#pragma once
#include <ATen/Config.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <torch/headeronly/util/AccumulateType.h>

// The acc_type/AccumulateType/AccumulateTypeDevice compile-time traits now
// live in torch/headeronly/util/AccumulateType.h (included above), which
// injects them into the `at` namespace. Only the runtime, non-header-only
// conversions from a `c10::ScalarType` to its accumulate `c10::ScalarType`
// remain here, since they are TORCH_API and require linking libtorch.

namespace at {

TORCH_API c10::ScalarType toAccumulateType(
    c10::ScalarType type,
    c10::DeviceType device);
TORCH_API c10::ScalarType toAccumulateType(c10::ScalarType type, bool is_cuda);

} // namespace at
