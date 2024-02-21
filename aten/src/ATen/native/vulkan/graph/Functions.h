#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/graph/Graph.h>

namespace at {
namespace native {
namespace vulkan {

ValueRef add(ComputeGraph& graph, const std::vector<ValueRef>& args);

ValueRef sub(ComputeGraph& graph, const std::vector<ValueRef>& args);

ValueRef mul(ComputeGraph& graph, const std::vector<ValueRef>& args);

ValueRef div(ComputeGraph& graph, const std::vector<ValueRef>& args);

ValueRef floor_div(ComputeGraph& graph, const std::vector<ValueRef>& args);

ValueRef pow(ComputeGraph& graph, const std::vector<ValueRef>& args);

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
