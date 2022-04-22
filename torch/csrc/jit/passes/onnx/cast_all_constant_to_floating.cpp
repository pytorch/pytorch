#include <torch/csrc/jit/passes/onnx/cast_all_constant_to_floating.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

namespace torch {
namespace jit {
namespace onnx {
using namespace ::c10::onnx;
}

// For ONNX opset < 9, constant operator supports only three data types:
// float16, float, and double. Constants of other data types are exported as
// float or double and then cast back to their original data type with a cast
// node. The above transformation is done in this pass. The motivation behind
// having it as a post process pass opposed to handling in symbolic, is that
// many constant operators would have already been removed in the export before
// this step. On the other hand if cast is inserted in symbolic, subsequent node
// conversion will break if it depends on certain inputs being constant.
void CastAllConstantToFloating(Block* block) {
  auto graph = block->owningGraph();
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      CastAllConstantToFloating(block);
    }

    if (node->kind() == onnx::Constant) {
      auto val = node->t(attr::value);
      at::ScalarType dtype = val.scalar_type();
      auto val_type = TensorType::create(val);
      if (dtype != at::ScalarType::Double && dtype != at::ScalarType::Float &&
          dtype != at::ScalarType::Half) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        int to_type;
        switch (val.scalar_type()) {
          case at::ScalarType::Byte:
          case at::ScalarType::Char:
          case at::ScalarType::Int:
          case at::ScalarType::Short:
          case at::ScalarType::Bool:
            to_type = ATenTypeToOnnxType(val.scalar_type());
            val = val.to(at::ScalarType::Float);
            break;

          case at::ScalarType::Long:
            to_type = ATenTypeToOnnxType(val.scalar_type());
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
        cast_node->output()->setType(val_type);
        cast_node->insertAfter(node);
        // get input from cast node
        node->outputs().at(0)->replaceAllUsesWith(cast_node->outputs().at(0));
        // add input from constant to cast node
        cast_node->addInput(node->outputs().at(0));
        cast_node->copyMetadata(node);
      }
    }
  }
}

void CastAllConstantToFloating(const std::shared_ptr<Graph>& graph) {
  CastAllConstantToFloating(graph->block());
}
} // namespace jit
} // namespace torch
