#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/graph/Graph.h>

namespace at {
namespace native {
namespace vulkan {

#define DEFINE_OP_FN(name) \
  ValueRef name(ComputeGraph& graph, const std::vector<ValueRef>& args);

DEFINE_OP_FN(add);
DEFINE_OP_FN(sub);
DEFINE_OP_FN(mul);
DEFINE_OP_FN(div);
DEFINE_OP_FN(floor_div);
DEFINE_OP_FN(pow);

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
