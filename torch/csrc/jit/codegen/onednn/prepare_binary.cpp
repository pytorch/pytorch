#include <aten/src/ATen/core/jit_type.h>
#include <torch/csrc/jit/codegen/onednn/prepare_binary.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/shape_analysis.h>

namespace torch::jit::fuser::onednn {

static bool compareConstValue(Value* v, double d) {
  auto ival = toIValue(v);
  return ival.has_value() &&
      ((ival->isInt() && static_cast<int>(ival->toInt()) == d) ||
       (ival->isDouble() && ival->toDouble() == d));
}

static void handleBinaryOpInputs(Node* node) {
  // We do not handle binary ops with two scalar inputs,
  // and we assume scalar is always at the second place.
  if (node->input(0)->type()->isSubtypeOf(TensorType::get())) {
    auto dtypeOfFirstInput =
        node->input(0)->type()->cast<TensorType>()->scalarType().value();
    if (node->input(1)->type()->isSubtypeOf(FloatType::get()) ||
        node->input(1)->type()->isSubtypeOf(IntType::get())) {
      // If a scalar is added to be a tensor, we would assume that the
      // scalar is of the same dtype as the tensor, as oneDNN graph
      // currently requires inputs of binary ops to have the same dtype.
      // We create a 1D tensor from the scalar input & "promote" its
      // dtype to that of the first input. Doing so helps us satisfy PyTorch's
      // type promotion rules.
      // Although we convert the scalar to a tensor, we still need to promote
      // types, as if the second input were still a scalar.
      // The following sample code-snippet illustrates that converting a scalar
      // input to a 1-D tensor may result in a different output dtype than would
      // otherwise have been the case.
      // clang-format off
      //   >>> (1. + torch.rand([2]).half()).dtype
      //       torch.float16
      //   >>> (torch.tensor(1.).unsqueeze(0) + (torch.rand([2]).half())).dtype
      //       torch.float32
      // clang-format on
      auto promotedDtype = dtypeOfFirstInput;
      auto scalar = node->input(1);
      WithInsertPoint guard(node);
      auto g = node->owningGraph();
      // 42 : Scalar  -->  tensor(42.0) : Float([])
      auto t = g->insert(aten::as_tensor, {scalar}, {{"dtype", promotedDtype}});
      // add dim & stride info to IR
      std::optional<size_t> t_dim = 1;
      auto target_type = TensorTypePtr(
          TensorType::create(promotedDtype, at::kCPU, t_dim, false));
      target_type = target_type->withSizes({1});
      t->setType(target_type);

      // tensor(42.0) : Float([])  -->  tensor([42.0]) : Float([1])
      auto unsqueezed = g->insert(aten::unsqueeze, {t, 0});
      unsqueezed->setType(target_type);
      node->replaceInput(1, unsqueezed);

      // dtype might have changed, so needs to be updated in IR as well
      node->output()->setType(
          node->output()->type()->expect<TensorType>()->withScalarType(
              promotedDtype));
    } else if (node->input(1)->type()->isSubtypeOf(TensorType::get())) {
      // Here, both inputs are tensors, and we just wanna make sure that they
      // are the same dtype, as oneDNN Graph requires both inputs to have the
      // same dtype. We'll follow PyTorch's type-promotion rules here.
      auto second_input_typeptr = node->input(1)->type()->expect<TensorType>();
      std::optional<at::ScalarType> second_input_type =
          second_input_typeptr->scalarType();
      if (second_input_type != std::nullopt) {
        // dtype of the second tensor might not be available in the IR
        auto dtypeOfSecondInput = second_input_type.value();
        if (dtypeOfFirstInput != dtypeOfSecondInput) {
          // Type promotion is required
          auto promotedDtype =
              c10::promoteTypes(dtypeOfFirstInput, dtypeOfSecondInput);
          WithInsertPoint guard(node);
          auto g = node->owningGraph();
          if (promotedDtype == dtypeOfFirstInput) {
            auto to_node_output = g->insert(
                aten::to, {node->input(1)}, {{"dtype", promotedDtype}});
            to_node_output->setType(
                node->input(1)->type()->expect<TensorType>()->withScalarType(
                    promotedDtype));
            node->replaceInput(1, to_node_output);
          } else {
            auto to_node_output = g->insert(
                aten::to, {node->input(0)}, {{"dtype", promotedDtype}});
            to_node_output->setType(
                node->input(0)->type()->expect<TensorType>()->withScalarType(
                    promotedDtype));
            node->replaceInput(0, to_node_output);
          }
          // dtype might have changed, so needs to be updated in IR as well
          node->output()->setType(
              node->output()->type()->expect<TensorType>()->withScalarType(
                  promotedDtype));
        } else {
          // both dtypes are same
          // IR info of dtypes is missing sometimes in JIT IR,
          // and we shouldn't treat those tensors as FP32 tensors by default.
          node->output()->setType(
              node->output()->type()->expect<TensorType>()->withScalarType(
                  dtypeOfFirstInput));
        }
      } // end inner if block
    } // end outer if block
  }
}

static void ConvertScalarToTensor(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      ConvertScalarToTensor(sub);
    }

    if (node->kind() == aten::add || node->kind() == aten::mul ||
        node->kind() == aten::div) {
      handleBinaryOpInputs(node);
    }
  }
}

static void mayDecomposeAdd(Node* node) {
  if (node->inputs().size() < 3) {
    return; // corner-case in BERT-mrpc that's not in line with
            // native_functions.yaml
  }
  if (toIValue(node->namedInput("alpha")).has_value()) {
    auto alphaEqualsOne = compareConstValue(node->namedInput("alpha"), 1.0);
    if (!alphaEqualsOne) {
      WithInsertPoint guard(node);
      auto g = node->owningGraph();
      auto mul = g->insert(
          aten::mul, {node->namedInput("other"), node->namedInput("alpha")});
      if (node->namedInput("other")->type()->isSubtypeOf(TensorType::get())) {
        auto mulTensorTypePtr = node->namedInput("other")->type();
        mul->setType(mulTensorTypePtr);
      }
      node->replaceInput(1, mul);
      auto one = g->insertConstant(1.0);
      node->replaceInput(2, one);
    }
  }
}

static void DecomposeFusedAdd(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      DecomposeFusedAdd(sub);
    }

    if (node->kind() == aten::add) {
      mayDecomposeAdd(node);
    }
  }
}

static void EliminateIdentityMulAdd(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      EliminateIdentityMulAdd(sub);
    }

    if ((node->kind() == aten::add && compareConstValue(node->input(1), 0.0)) ||
        (node->kind() == aten::mul && compareConstValue(node->input(1), 1.0))) {
      node->output()->replaceAllUsesWith(node->namedInput("self"));
    }
  }
}

void PrepareBinaryForLLGA(const std::shared_ptr<Graph>& graph) {
  DecomposeFusedAdd(graph->block());
  EliminateIdentityMulAdd(graph->block());
  EliminateDeadCode(graph);
  // ConvertScalarToTensor must be placed after EliminateIdentityMulAdd
  ConvertScalarToTensor(graph->block());
}

} // namespace torch::jit::fuser::onednn
