#include <stack>

#include <ATen/core/jit_type.h>
#include <ATen/native/xnnpack/OpContext.h>

#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/xnnpack_rewrite.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

namespace {

#ifdef USE_XNNPACK
using at::native::xnnpack::XNNPackLinearOpContext;
using at::native::xnnpack::XNNPackConv2dOpContext;

bool nodeMatchesPackingOps(Node* n) {
  return ((n->kind() == Symbol::fromQualString("xnnpack::linear_prepack")) ||
      n->kind() == Symbol::fromQualString("xnnpack::conv2d_prepack"));
}

// Must run this pass after constant folding.
void removePrePackingOps_(script::Module& m) {
  auto method = m.get_method("forward");
  auto graph = method.graph();
  std::stack<Block*> blocks_to_visit;
  std::unordered_set<Node*> nodes_to_delete;
  blocks_to_visit.push(graph->block());
  int64_t uid = 0;
  std::string attr_name_base("packed_weight_");
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      if (nodeMatchesPackingOps(n)) {
        auto optional_outputs = runNodeIfInputsAreConstant(n);
        if (optional_outputs) {
          auto outputs = optional_outputs.value();
          TORCH_CHECK(outputs.size() == 1, "Prepack ops have single output");
          auto attr_name = attr_name_base + c10::to_string(uid++);
          m.register_attribute(attr_name, n->outputs()[0]->type(), outputs[0]);
          Value* prepack_op_value = n->outputs()[0];
          WithInsertPoint ins(prepack_op_value->node());
          Value* packed_weight_attr =
            graph->insertGetAttr(graph->inputs()[0], attr_name)
                  ->setType(n->outputs()[0]->type());
          prepack_op_value->replaceAllUsesWith(packed_weight_attr);
          //n->removeAllInputs(); //Cannot do this for conv because it will remove constant nodes?
          nodes_to_delete.insert(n);
        }
      }
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  for (auto n : nodes_to_delete) {
    n->removeAllInputs();
  }
  for (auto n : nodes_to_delete) {
    n->destroy();
  }
}

void insertXNNPACKLinearOp(std::shared_ptr<Graph>& graph) {
  std::string linear_before_inline = R"(
    graph(%linear, %input, %weight, %bias):
        %r = prim::CallFunction(%linear, %input, %weight, %bias)
        return (%r))";
  std::string xnnpack_pattern_before_inline = R"(
    graph(%linear, %input, %weight, %bias):
        %packed_weight_bias = xnnpack::linear_prepack(%weight, %bias)
        %res = xnnpack::linear_packed(%input, %packed_weight_bias)
        return (%res))";
  std::string linear_pattern = R"(
    graph(%input, %weight, %bias):
        %r = aten::linear(%input, %weight, %bias)
        return (%r))";
  std::string xnnpack_pattern = R"(
    graph(%input, %weight, %bias):
        %packed_weight_bias = xnnpack::linear_prepack(%weight, %bias)
        %res = xnnpack::linear_packed(%input, %packed_weight_bias)
        return (%res))";

  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto linear_value = match_vmap.at(vmap.at("linear"));
    auto func_name = graph_rewrite_helper::getFuncName(linear_value);
    if (func_name == "linear") {
      return true;
    }
    return false;
  };

  SubgraphRewriter linear_call_fn_rewriter;
  linear_call_fn_rewriter.RegisterRewritePattern(linear_before_inline, xnnpack_pattern_before_inline);
  linear_call_fn_rewriter.runOnGraph(graph, filter);

  SubgraphRewriter linear_rewriter;
  linear_rewriter.RegisterRewritePattern(linear_pattern, xnnpack_pattern);
  linear_rewriter.runOnGraph(graph);
}

void insertXNNPACKConv2dOp(std::shared_ptr<Graph>& graph) {
  // Replace _convolution with conv2d
  graph_rewrite_helper::replaceConvolutionWithConv2d(graph);

  std::string conv_2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = aten::conv2d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string xnnpack_conv2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %packed_weight_bias = xnnpack::conv2d_prepack(%weight, %bias, %stride, %padding, %dilation, %groups)
        %r = xnnpack::conv2d_packed(%input, %packed_weight_bias)
        return (%r) )";

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(conv_2d_pattern, xnnpack_conv2d_pattern);
  rewriter.runOnGraph(graph);
}

} // namespace


void insertXNNPACKOps(std::shared_ptr<Graph>& graph) {
  ConstantPooling(graph);
  ConstantPropagation(graph);
  insertXNNPACKLinearOp(graph);
  insertXNNPACKConv2dOp(graph);
}

void insertXNNPACKOps(script::Module& module) {
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    insertXNNPACKOps(graph);
  }
  for (script::Module m : module.children()) {
    insertXNNPACKOps(m);
  }
}

void removePrePackingOps(script::Module& m) {
  removePrePackingOps_(m);
}

#else

void insertXNNPACKOps(std::shared_ptr<Graph>& graph) {
  TORCH_INTERNAL_ASSERT("XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

void insertXNNPACKOps(script::Module& module) {
  TORCH_INTERNAL_ASSERT("XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

void removePrePackingOps(script::Module& m) {
  TORCH_INTERNAL_ASSERT("XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

#endif
} // namespace jit
} // namespace torch
