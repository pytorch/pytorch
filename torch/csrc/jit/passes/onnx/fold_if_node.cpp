#include <torch/csrc/jit/passes/onnx/fold_if_node.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/onnx/scalar_type_analysis.h>
#include <torch/torch.h>

#include <c10/util/Optional.h>
#include <algorithm>

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

static bool checkIfFold(Node* node, bool dynamic_axes) {
  if (node->kind() != onnx::If)
    return false;

  auto cast_node = node->input()->node();
  // assumes that Cast node is always before If node,
  // but sometimes that is not the case.
  if (cast_node->kind() != onnx::Cast)
    return false;
  auto prev_node = cast_node->input()->node();
  
  Node* compare_node = nullptr;
  if (prev_node->kind() == onnx::Not || prev_node->kind() == onnx::Identity) {
    compare_node = prev_node->input()->node();
  } else {
    compare_node = cast_node->input()->node();
  }
  if (compare_node->kind() == onnx::If) {
    return checkIfFold(compare_node, dynamic_axes);
  }
  if (compare_node->kind() == onnx::Equal ||
      compare_node->kind() == onnx::Greater ||
      compare_node->kind() == onnx::Less) {
    for (size_t i = 0; i < compare_node->inputs().size(); i++) {
      if (!(compare_node->inputs()[i]->node()->kind() == onnx::Constant ||
            compare_node->inputs()[i]->node()->kind() == onnx::Size ||
            compare_node->inputs()[i]->node()->kind() == onnx::ReduceProd))
        return false;
      if (compare_node->inputs()[i]->node()->kind() == onnx::ReduceProd &&
          dynamic_axes)
        return false;
    }
    return true;
  } else if (compare_node->kind() == onnx::Constant) {
    return true;
  }
  return false;
}

static bool constantFoldingValue(Node* node) {
  auto cast_node = node->input()->node();
  auto prev_node = cast_node->input()->node();
  if (prev_node->kind() == onnx::If) {
    int cond = 1 - (int)constantFoldingValue(prev_node);
    Block* block = prev_node->blocks()[cond];
    // we are assuming that the node will be Constant and
    // that number of block outputs is 1.
    prev_node = block->outputs()[0]->node();
  }

  if (prev_node->kind() == onnx::Constant) {
    const at::Tensor val = prev_node->t(attr::value);
    return at::is_nonzero(val);
  }

  if (prev_node->kind() == onnx::Identity &&
      prev_node->input()->node()->kind() == onnx::Constant) {
    auto val = prev_node->input()->node()->t(attr::value);
    return at::is_nonzero(val);
  }

  Node* compare_node = nullptr;
  if (prev_node->kind() == onnx::Not) {
    compare_node = prev_node->input()->node();
  } else if (cast_node->inputs().size() > 0) {
    compare_node = cast_node->input()->node();
  }
  TORCH_INTERNAL_ASSERT(compare_node != nullptr);
  ScalarTypeAnalysisNodeForONNX(compare_node);
  std::vector<at::Tensor> inputs;
  for (size_t i = 0; i < compare_node->inputs().size(); i++) {
    auto input_node = compare_node->inputs()[i]->node();
    if (input_node->kind() == onnx::Constant) {
      const at::Tensor val = input_node->t(attr::value);
      inputs.push_back(val);
    } else { // input_node is either onnx::Size or onnx::ReduceProd
      auto shape_node = input_node->input()->node();
      auto shape =
          shape_node->input()->type()->cast<TensorType>()->symbolic_sizes();

      at::Tensor val;
      if (input_node->kind() == onnx::Size) {
        auto rank = shape.rank();
        val = c10::scalar_to_tensor((int64_t)*rank);
      } else if (input_node->kind() == onnx::ReduceProd) {
        auto sizes = shape.sizes();
        int64_t prod = 1;
        for (int64_t i = 0; i < (int64_t)*shape.rank(); i++) {
          auto dim = sizes.value()[i].static_size();
          prod *= dim;
        }
        val = c10::scalar_to_tensor(prod);
      }

      inputs.push_back(val);
    }
  }

  if (compare_node->kind() == onnx::Equal) {
    if (prev_node->kind() == onnx::Not)
      return !(at::equal(inputs[0], inputs[1]));
    return at::equal(inputs[0], inputs[1]);
  } else if (
      (compare_node->kind() == onnx::Greater &&
       prev_node->kind() != onnx::Not) ||
      (prev_node->kind() == onnx::Not && compare_node->kind() == onnx::Less)) {
    auto res = at::greater(inputs[0], inputs[1]);
    if (prev_node->kind() == onnx::Not)
      res = at::greater_equal(inputs[0], inputs[1]);
    return at::is_nonzero(res);
  } else if (
      compare_node->kind() == onnx::Less ||
      (prev_node->kind() == onnx::Not &&
       compare_node->kind() == onnx::Greater)) {
    auto res = at::less(inputs[0], inputs[1]);
    if (prev_node->kind() == onnx::Not)
      res = at::less_equal(inputs[0], inputs[1]);
    return at::is_nonzero(res);
  }
  return false;
}

// before peephole pass
// graph(%y.2 : Int(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
//   %4 : Long(2, strides=[1], device=cpu) = onnx::Shape(%y.2)
//   %5 : Long(device=cpu) = onnx::Size(%4)
//   %12 : Long(requires_grad=0, device=cpu) = onnx::Constant[value={2}]()
//   %6 : Bool(device=cpu) = onnx::Equal(%5, %12)
//   %11 : bool = onnx::Cast[to=9](%6)
//   %7 : Int(3, 4, strides=[4, 1], device=cpu) = onnx::If(%11)
//     block0():
//       %13 : Int(requires_grad=0, device=cpu) = onnx::Constant[value={4}]()
//       %8 : Int(3, 4, strides=[4, 1], device=cpu) = onnx::Add(%y.2, %13)
//       %14 : Int(requires_grad=0, device=cpu) = onnx::Constant[value={2}]()
//       %9 : Int(3, 4, strides=[4, 1], device=cpu) = onnx::Add(%8, %14)
//       -> (%9)
//     block1():
//       %y.1 : Int(3, 4, strides=[4, 1], requires_grad=0, device=cpu) =
//       onnx::Identity(%y.2)
//       -> (%y.1)
//   return (%7)

// after peephole pass
// graph(%y.2 : Int(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
//   %4 : Long(2, strides=[1], device=cpu) = onnx::Shape(%y.2)
//   %5 : Long(device=cpu) = onnx::Size(%4)
//   %12 : Long(requires_grad=0, device=cpu) = onnx::Constant[value={2}]()
//   %6 : Bool(device=cpu) = onnx::Equal(%5, %12)
//   %11 : bool = onnx::Cast[to=9](%6)
//   %13 : Int(requires_grad=0, device=cpu) = onnx::Constant[value={4}]()
//   %8 : Int(3, 4, strides=[4, 1], device=cpu) = onnx::Add(%y.2, %13)
//   %14 : Int(requires_grad=0, device=cpu) = onnx::Constant[value={2}]()
//   %9 : Int(3, 4, strides=[4, 1], device=cpu) = onnx::Add(%8, %14)
//   return (%9)

static void foldIfNode(Block* b, bool dynamic_axes) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      foldIfNode(child_block, dynamic_axes);
    }
    if (it->kind() == onnx::If) {
      auto if_node = *it;
      if (checkIfFold(if_node, dynamic_axes)) {
        Block* then_block = it->blocks()[0];
        Block* else_block = it->blocks()[1];
        Block* block = else_block;
        if (constantFoldingValue(if_node))
          block = then_block;

        std::vector<Node*> nodes_in_valid_path;
        for (auto* valid_node : block->nodes()) {
          nodes_in_valid_path.push_back(valid_node);
        }
        Node* cur = if_node;
        for (auto* valid_node : nodes_in_valid_path) {
          valid_node->moveAfter(cur);
          cur = valid_node;
        }
        for (size_t i = 0; i < block->return_node()->inputs().size(); ++i) {
          if_node->outputs()[i]->replaceAllUsesWith(
              block->return_node()->inputs()[i]);
        }
        it->removeAllInputs();
        it.destroyCurrent();
      }
    }
  }
}


  void FoldIfONNX(Block * b, bool dynamic_axes) {
    foldIfNode(b, dynamic_axes);
  }

  bool FoldConditionONNX(Node * n) {
    return constantFoldingValue(n);
  }

  bool CheckFoldONNX(Node * n) {
    return checkIfFold(n, false);
  }

} // namespace jit
} // namespace jit
