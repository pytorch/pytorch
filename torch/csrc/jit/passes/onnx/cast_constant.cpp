#include <torch/csrc/jit/passes/onnx/cast_constant.h>

namespace torch {
namespace jit {
namespace onnx {
using namespace ::c10::onnx;
}

void CastConstant(Block* block) {
  auto graph = block->owningGraph();
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      CastConstant(block);
    }

    if (node->kind() == onnx::Constant) {
      auto val = node->t(attr::value);
      at::ScalarType dtype = val.scalar_type();
      if (dtype != at::ScalarType::Double && dtype != at::ScalarType::Float && dtype != at::ScalarType::Half) {
        int to_type;
        switch (val.scalar_type()){
          case at::ScalarType::Byte:
          case at::ScalarType::Char:
          case at::ScalarType::Int:
          case at::ScalarType::Short:
          case at::ScalarType::Bool:
            to_type = 6; // Int32
            val = val.to(at::ScalarType::Float);
            break;

          case at::ScalarType::Long:
            to_type = 7; // Int64
            val = val.to(at::ScalarType::Double);
            break;

          default:
            throw std::runtime_error("Unsupported types: complex, string");
        }
        // create a cast node
        node->removeAttribute(attr::value);
        node->t_(attr::value, val);
        Node* cast_node = graph->create(onnx::Cast, 1);
        cast_node->i_(attr::to, to_type);
        cast_node->insertAfter(node);
        // get input from cast node
        node->outputs().at(0)->replaceAllUsesWith(cast_node->outputs().at(0));
        // add input from constant to cast node
        cast_node->addInput(node->outputs().at(0));
      }

    }
  }
}

void CastConstant(const std::shared_ptr<Graph>& graph) {
  CastConstant(graph->block());
}
} // namespace jit
} // namespace torch
