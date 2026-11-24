#pragma once
#include <ATen/Config.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <torch/headeronly/core/AccumulateType.h>

namespace at {

using torch::headeronly::acc_type;
using torch::headeronly::acc_type_device;
using torch::headeronly::AccumulateType;
using torch::headeronly::AccumulateTypeDevice;

TORCH_API c10::ScalarType toAccumulateType(
    c10::ScalarType type,
    c10::DeviceType device);
TORCH_API c10::ScalarType toAccumulateType(c10::ScalarType type, bool is_cuda);

} // namespace at
