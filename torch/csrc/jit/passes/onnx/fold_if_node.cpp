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

// This function determines wheather If Node can be folded.
static bool isStaticCondition(Node* node) {
  TORCH_INTERNAL_ASSERT(
      node->kind() == onnx::If || node->kind() == onnx::Not ||
      node->kind() == onnx::Identity);
  auto cast_node = node->input()->node();
  if (cast_node->kind() != onnx::Cast)
    cast_node = node;
  auto prev_node = cast_node->input()->node();

  if (prev_node->kind() == onnx::Not || prev_node->kind() == onnx::Identity ||
      prev_node->kind() == onnx::If)
    return isStaticCondition(prev_node);

  auto compare_node = prev_node;
  if (compare_node->kind() == onnx::Equal ||
      compare_node->kind() == onnx::Greater ||
      compare_node->kind() == onnx::Less ||
      compare_node->kind() == onnx::GreaterOrEqual ||
      compare_node->kind() == onnx::LessOrEqual) {
    for (size_t i = 0; i < compare_node->inputs().size(); i++) {
      auto sym = compare_node->inputs()[i]
                     ->type()
                     ->castRaw<TensorType>()
                     ->symbolic_sizes();
      if (!(compare_node->inputs()[i]->node()->kind() == onnx::Constant ||
            compare_node->inputs()[i]->node()->kind() == onnx::Size ||
            compare_node->inputs()[i]->node()->kind() == onnx::ReduceProd))
        return false;
      if (compare_node->inputs()[i]->node()->kind() != onnx::Constant) {
        auto shape_node = compare_node->inputs()[i]->node()->input()->node();
        auto shape = shape_node->input()
                         ->type()
                         ->castRaw<TensorType>()
                         ->symbolic_sizes();

        // ONNX shape and type inference cannot determine the shape of the input
        if (!shape.rank())
          return false;

        // If dynamic_axes are used on inputs to ReduceProd node, don't fold If
        // node
        auto dynamic_axes = shape.isComplete();
        if (!dynamic_axes &&
            compare_node->inputs()[i]->node()->kind() == onnx::ReduceProd)
          return false;
      }
    }
    return true;
  } else if (compare_node->kind() == onnx::Constant) {
    return true;
  }
  return false;
}

// find index of the block output
static c10::optional<int> findIndex(
    c10::ArrayRef<torch::jit::Value*> outputs,
    Value* input) {
  c10::optional<int> idx = c10::nullopt;
  for (size_t i = 0; i < outputs.size(); i++) {
    if (input == outputs[i]) {
      idx = i;
      break;
    }
  }
  return idx;
}

// This function returns the value of the constant-folded subblock
// that is input to the If node.
static bool constantFoldedConditionValue(Node* node) {
  TORCH_INTERNAL_ASSERT(node->kind() == onnx::If);
  // usually Cast node precedes If node in the graph, but
  // there are some rare scenarios when that is not the case.
  auto cast_node = node->input()->node();
  if (cast_node->kind() != onnx::Cast)
    cast_node = node;
  auto prev_node = cast_node->input()->node();
  if (prev_node->kind() == onnx::If) {
    int cond = 1 - (int)constantFoldedConditionValue(prev_node);
    Block* block = prev_node->blocks()[cond];
    auto outputs = cast_node->input()->node()->outputs();
    auto cast_input = cast_node->input();
    int idx = findIndex(outputs, cast_input).value();
    prev_node = block->outputs()[idx]->node();
  }

  if (prev_node->kind() == onnx::Constant) {
    const at::Tensor& val = prev_node->t(attr::value);
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
      const at::Tensor& val = input_node->t(attr::value);
      inputs.push_back(val);
    } else { // input_node is either onnx::Size or onnx::ReduceProd
      auto shape_node = input_node->input()->node();
      auto shape =
          shape_node->input()->type()->castRaw<TensorType>()->symbolic_sizes();

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

  at::Tensor res;
  if (compare_node->kind() == onnx::Equal) {
    res = at::eq(inputs[0], inputs[1]);
    if (prev_node->kind() == onnx::Not)
      res = at::not_equal(inputs[0], inputs[1]);
  } else if (
      compare_node->kind() == onnx::Greater && prev_node->kind() != onnx::Not) {
    res = at::greater(inputs[0], inputs[1]);
  } else if (
      (prev_node->kind() == onnx::Not && compare_node->kind() == onnx::Less) ||
      compare_node->kind() == onnx::GreaterOrEqual) {
    res = at::greater_equal(inputs[0], inputs[1]);
  } else if (
      compare_node->kind() == onnx::Less && prev_node->kind() != onnx::Not) {
    res = at::less(inputs[0], inputs[1]);
  } else if (
      (prev_node->kind() == onnx::Not &&
       compare_node->kind() == onnx::Greater) ||
      compare_node->kind() == onnx::LessOrEqual) {
    res = at::less_equal(inputs[0], inputs[1]);
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Condition value of the If node could not be constant-folded!");
  }

  return at::is_nonzero(res);
}

// This pass return then or else branch of the If node depending on the
// value of the constant-folded sublock that is input to the If node
//
// Example:
// before post pass
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

// after post pass
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

static void foldIfNode(Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      foldIfNode(child_block);
    }
    if (it->kind() == onnx::If) {
      auto if_node = *it;
      if (isStaticCondition(if_node)) {
        Block* then_block = it->blocks()[0];
        Block* else_block = it->blocks()[1];
        Block* block = else_block;
        if (constantFoldedConditionValue(if_node))
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

// This pass is folding If node when the condition (subblock) can be
// constant-folded. Currently ONNX Runtime is doing Shape and Type Inference on
// both branches of the If operator, regardless of which branch is executing in
// Runtime. This can cause runtime errors in some cases:
// 1. Condition of the If node is based on shape / size of the input
// 2. then and else branch have different return types
// Folding If node can prevent Runtime errors in ONNXRuntime.
void FoldIfNodeONNX(Block* b) {
  foldIfNode(b);
}

bool ConditionValueONNX(Node* n) {
  return constantFoldedConditionValue(n);
}

bool IsStaticConditionONNX(Node* n) {
  return isStaticCondition(n);
}

} // namespace jit
} // namespace torch
