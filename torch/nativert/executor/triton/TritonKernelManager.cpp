#include <torch/nativert/executor/triton/TritonKernelManager.h>

#include <c10/util/Exception.h>

namespace torch::nativert {

void LaunchParams::parseCommonAttributes(const Node* node) {
  for (const auto& attr : node->attributes()) {
    std::vector<int64_t> grid;
    if (set_from_variant<std::vector<int64_t>>(grid, "grid", attr)) {
      TORCH_CHECK(grid.size() == 3, "grid must be a 3D vector");
      grid_dims = GridDims(
          static_cast<int>(grid[0]),
          static_cast<int>(grid[1]),
          static_cast<int>(grid[2]));
    }
  }
}

std::unique_ptr<LaunchParams> TritonKernelManager::createLaunchParams(
    const Node* node) const {
  auto params = std::make_unique<LaunchParams>();
  params->parseCommonAttributes(node);
  return params;
}

} // namespace torch::nativert
