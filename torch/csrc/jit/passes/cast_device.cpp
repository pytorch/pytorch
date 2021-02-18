#include <torch/csrc/jit/passes/cast_device.h>

namespace torch {
namespace jit {

namespace {
void castToDeviceImpl(Block* block, c10::Device device) {
  std::stringstream ss;
  ss << device;

  for (auto it = block->nodes().rbegin(); it != block->nodes().rend(); it++) {
    Node* node = *it;
    for (auto block : node->blocks()) {
      castToDeviceImpl(block, device);
    }

    if (node->kind() == prim::Constant && (*node->output()->type() == *DeviceObjType::get())) {
      node->s_(attr::value, ss.str());
    }
  }
}
} // namespace

void castToDevice(std::shared_ptr<Graph>& graph, c10::Device device) {
  castToDeviceImpl(graph->block(), device);
}

void castToDevice(script::Module& module, c10::Device device) {
  // "Cast" appropriate nodes in method graphs to device.
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    castToDevice(graph, device);
  }

  // Move module buffers and parameters to device.
  module.to(device);
}

} // namespace jit
} // namespace torch
