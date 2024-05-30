#pragma once

#include <ATen/core/Tensor.h>

namespace torch {
namespace utils {

TORCH_API void check_out_type_matches(
    const at::Tensor& result,
    std::optional<at::ScalarType> scalarType,
    bool scalarType_is_none,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    bool device_is_none);

}
} // namespace torch
