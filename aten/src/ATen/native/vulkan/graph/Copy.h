#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/graph/Graph.h>

namespace at {
namespace native {
namespace vulkan {

class CopyNode : public virtual OpNode {
 public:
  explicit CopyNode(ValueRef from, ValueRef to);

  void encode(ComputeGraph* graph);
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
