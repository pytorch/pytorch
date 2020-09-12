#pragma once

#include <ATen/ATen.h>
#include <ATen/native/vulkan/api/api.h>
#include <ATen/native/vulkan/ops/Tensor.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace details {

template<typename From, typename To>
using is_convertible = std::enable_if_t<std::is_convertible<From, To>::value>;

template<typename Pointer>
using is_pointer = std::enable_if_t<std::is_pointer<Pointer>::value>;

} // namespace details
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
