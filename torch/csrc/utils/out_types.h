#pragma once

#include <ATen/core/Tensor.h>

namespace torch {
namespace utils {

TORCH_API void check_out_type_matches(
    const at::Tensor& result,
    at::ScalarType scalarType, bool scalarType_is_none,
    c10::optional<at::Layout> layout,
    const at::Device& device, bool device_is_none);

}}
