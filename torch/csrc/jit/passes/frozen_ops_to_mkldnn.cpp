#include <ATen/Utils.h>

#include <ATen/Config.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/frozen_conv_folding.h>
#include <torch/csrc/jit/passes/frozen_ops_to_mkldnn.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <torch/csrc/jit/tensorexpr/types.h>

#include <ATen/core/interned_strings.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

namespace torch {
namespace jit {

using Tensor = at::Tensor;

namespace {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return AliasAnalysisKind::FROM_SCHEMA;
}

// These operators are registered as builtins instead of
// using control flow because it will make it easier to
// remove unneeded adjacent mkldnn/dense conversions
RegisterOperators mm_tree_reduction_reg(
    {Operator(
         "prim::ConvertToMKLDNN(Tensor input) -> (bool, Tensor)",
         [](Stack* stack) {
           auto input = pop(stack).toTensor();
           bool is_mkldnn = input.is_mkldnn();
           push(stack, is_mkldnn);
           if (input.is_mkldnn()) {
             push(stack, input);
           } else {
             push(stack, input.to_mkldnn());
           }
         },
         // registered as a special case so that it can return two outputs
         // instead of a tuple tuple unboxing wouldn't be able to be removed and
         // gets in the way of transformation
         AliasAnalysisKind::INTERNAL_SPECIAL_CASE),
     Operator(
         "prim::ConvertFromMKLDNN(bool to_mkldnn, Tensor(a) input) -> (Tensor(a))",
         [](Stack* stack) {
           auto input = pop(stack).toTensor();
           bool to_mkldnn = pop(stack).toBool();
           if (to_mkldnn) {
             if (input.is_mkldnn()) {
               push(stack, input);
             } else {
               push(stack, input.to_mkldnn());
             }
           } else {
             if (input.is_mkldnn()) {
               push(stack, input.to_dense());
             } else {
               push(stack, input);
             }
           }
         },
         aliasAnalysisFromSchema())});

Operation ConstantMKLDNNTensorOp(const Node* node) {
  auto t = node->t(attr::value);
  return [t](Stack* stack) {
    push(stack, t);
    return 0;
  };
}

// This is registered as its own op instead of as prim::Constant bc it does not
// serialize which is an invariant of prim::Consstant
RegisterOperators MKLDNNConstantOp({
    torch::jit::Operator(
        prim::ConstantMKLDNNTensor,
        ConstantMKLDNNTensorOp,
        AliasAnalysisKind::INTERNAL_SPECIAL_CASE),
});

Node* createConstantMKLDNNTensorOp(Graph* g, Tensor mkldnn_tensor) {
  TORCH_INTERNAL_ASSERT(mkldnn_tensor.is_mkldnn());
  auto op = g->create(prim::ConstantMKLDNNTensor);
  op->t_(attr::value, mkldnn_tensor);
  return op;
}

void computeOpInMKLDNN(Node* n) {
  auto graph = n->owningGraph();

  auto to_mkldnn =
      graph->create(Symbol::prim("ConvertToMKLDNN"), 2)->insertBefore(n);
  to_mkldnn->addInput(n->input(0));
  Value* was_mkldnn = to_mkldnn->outputs().at(0)->setType(BoolType::get());
  Value* mkldnn_tensor = to_mkldnn->outputs().at(1)->setType(TensorType::get());

  n->replaceInput(0, mkldnn_tensor);

  auto from_mkldnn =
      graph
          ->create(Symbol::prim("ConvertFromMKLDNN"), {was_mkldnn, n->output()})
          ->insertAfter(n);
  n->output()->replaceAllUsesAfterNodeWith(from_mkldnn, from_mkldnn->output());
}

bool nonConstantParameters(Node* n) {
  for (size_t i = 1; i < n->inputs().size(); i++) {
    if (n->inputs().at(i)->node()->kind() != prim::Constant) {
      return true;
    }
  }
  return false;
}

bool frozenMkldnnCompatibleConvNode(Node* n) {
  if (nonConstantParameters(n)) {
    return false;
  }
  // mkldnn does not support conv1d
  // _convolution is rewritten before this pass is invoked
  if (n->kind() != aten::conv2d && n->kind() != aten::conv3d) {
    return false;
  }

  auto weight = constant_as<Tensor>(n->namedInput("weight")).value();
  if (!weight.device().is_cpu()) {
    return false;
  }
  // only supported mkldnn dtype for conv
  if (weight.dtype() != c10::ScalarType::Float) {
    return false;
  }
  return true;
}

void ConvertFrozenConvParamsToMKLDNN(Block* b) {
  auto g = b->owningGraph();
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      ConvertFrozenConvParamsToMKLDNN(block);
    }

    if (frozenMkldnnCompatibleConvNode(n)) {
      auto conv = n;
      auto conv_w_mkldnn =
          constant_as<Tensor>(conv->namedInput("weight")).value().to_mkldnn();
      std::vector<int64_t> padding =
          toIValue(conv->namedInput("padding"))->toIntVector();
      std::vector<int64_t> stride =
          toIValue(conv->namedInput("stride"))->toIntVector();
      std::vector<int64_t> dilation =
          toIValue(conv->namedInput("dilation"))->toIntVector();
      auto groups = constant_as<int64_t>(conv->namedInput("groups")).value();

      if (n->kind() == aten::conv2d) {
        conv_w_mkldnn = mkldnn_reorder_conv2d_weight(
            conv_w_mkldnn, padding, stride, dilation, groups);
      } else if (n->kind() == aten::conv3d) {
        conv_w_mkldnn = mkldnn_reorder_conv3d_weight(
            conv_w_mkldnn, padding, stride, dilation, groups);
      } else {
        TORCH_INTERNAL_ASSERT(false);
      }

      WithInsertPoint guard(conv);
      auto conv_w_mkldnn_value = createConstantMKLDNNTensorOp(g, conv_w_mkldnn)
                                     ->insertBefore(conv)
                                     ->output();
      auto conv_w_value = conv->namedInput("weight");
      conv_w_mkldnn_value->setDebugName(conv_w_value->debugName() + "_mkldnn");
      conv->replaceInputWith(conv_w_value, conv_w_mkldnn_value);

      if (conv->namedInput("bias")->type() != NoneType::get()) {
        auto conv_bias_mkldnn =
            constant_as<Tensor>(conv->namedInput("bias"))->to_mkldnn();
        auto conv_bias_mkldnn_value =
            createConstantMKLDNNTensorOp(g, conv_bias_mkldnn)
                ->insertBefore(conv)
                ->output();
        auto conv_bias_value = conv->namedInput("bias");
        conv_bias_mkldnn_value->setDebugName(
            conv_bias_value->debugName() + "_mkldnn");
        conv->replaceInputWith(conv_bias_value, conv_bias_mkldnn_value);
      }
      computeOpInMKLDNN(conv);
    }
  }
}

} // namespace

void ConvertFrozenOpsToMKLDNN(std::shared_ptr<Graph>& graph) {
#if AT_MKLDNN_ENABLED()
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);
  ConvertFrozenConvParamsToMKLDNN(graph->block());
  EliminateDeadCode(graph);
#endif
}

} // namespace jit
} // namespace torch
