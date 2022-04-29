#include <ATen/Config.h>
#include <ATen/code_template.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/mkldnn_rewrite.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

namespace torch {
namespace jit {

namespace mkldnn {
namespace detail {

#if AT_MKLDNN_ENABLED()
bool mkldnn_fuser_enabled = true;
#else
bool mkldnn_fuser_enabled = false;
#endif // AT_MKLDNN_ENABLED()

} // namespace detail
} // namespace mkldnn

#if AT_MKLDNN_ENABLED()

c10::VaryingShape<int64_t> getSizesOf(Node* n, size_t idx) {
  auto tt = n->input(idx)->type()->cast<TensorType>();
  return tt->sizes();
}

bool ndimEquals(c10::VaryingShape<int64_t> sizes, size_t value) {
  if (!sizes.concrete_sizes()) {
    return false;
  }
  auto concrete_sizes = *sizes.concrete_sizes();
  if (concrete_sizes.size() != value) {
    return false;
  }
  return true;
}

void insertPrePackedConvOpForNode(Node* n) {
  // Need to know all the shapes of input and weight
  auto input_sizes = getSizesOf(n, /*idx*/ 0);
  if (!ndimEquals(input_sizes, 4)) {
    return;
  }

  auto weight_sizes = getSizesOf(n, /*idx*/ 1);
  if (!ndimEquals(weight_sizes, 4)) {
    return;
  }

  WithInsertPoint guard(n);
  auto graph = n->owningGraph();

  IValue input_size_value(*input_sizes.concrete_sizes());
  auto input_size = graph->insertConstant(input_size_value);

  auto prepack_node = graph->create(
      Symbol::fromQualString("mkldnn_prepacked::conv2d_prepack"), 1);

  // skip input value
  for (auto i = 1; i < n->inputs().size(); i++) {
    Value* v = n->input(i);
    prepack_node->addInput(v);
  }
  prepack_node->addInput(input_size);
  auto attr = graph->insertConstant(IValue("none"));
  prepack_node->addInput(attr);
  prepack_node->output()->setType(
      getCustomClass("__torch__.torch.classes.mkldnn.Conv2dOpContext"));
  graph->insertNode(prepack_node);

  auto prepack_conv = graph->insertNode(
      graph->create(Symbol::fromQualString("mkldnn_prepacked::conv2d_run"), 1));
  prepack_conv->addInput(n->input(0));
  prepack_conv->addInput(prepack_node->output());
  prepack_conv->output()->setType(n->output()->type()->cast<TensorType>());

  n->output()->replaceAllUsesWith(prepack_conv->output());
}

bool canFuseOnDevice(Value* v) {
  auto type = v->type()->cast<TensorType>();
  if (!type) {
    return true;
  }
  auto device = type->device();
  if (!device) {
    return false;
  }
  return device->is_cpu() && mkldnn::detail::mkldnn_fuser_enabled;
}

bool isFusableOnDevice(Node* node) {
  for (const auto& input : node->inputs()) {
    if (!canFuseOnDevice(input)) {
      return false;
    }
  }
  return true;
}

void insertPrePackedConvOp(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* b : n->blocks()) {
      insertPrePackedConvOp(b);
    }

    if (n->kind() == aten::conv2d) {
      if (isFusableOnDevice(n)) {
        insertPrePackedConvOpForNode(n);
      }
    }
  }
  EliminateDeadCode(b);
}

void insertMkldnnPrePackedConv2dOp(std::shared_ptr<Graph>& graph) {
  // Replace _convolution with conv2d
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  insertPrePackedConvOp(graph->block());
}

void insertMkldnnPrePackedOps(std::shared_ptr<Graph>& graph) {
  insertMkldnnPrePackedConv2dOp(graph);
}

void insertMkldnnPrePackedOps(script::Module& module) {
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    insertMkldnnPrePackedOps(graph);
  }
  for (script::Module m : module.children()) {
    insertMkldnnPrePackedOps(m);
  }
}

// Eltwise inplace OP shares the same op_attr
std::string PrepareAttr(const std::string& op) {
  auto pos = op.find("_");
  if (pos != std::string::npos) {
    return op.substr(0, pos);
  } else {
    return op;
  }
}

void FuseReluWithPackedOps(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;
  std::vector<std::string> fusion_operators = {"relu"};
  auto conv_op_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %input_size:int[], %dummy_attr:str):
        %packed_weight_bias = mkldnn_prepacked::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %input_size, %dummy_attr)
        %conv2d_res = mkldnn_prepacked::conv2d_run(%input, %packed_weight_bias)
        %res = aten::${op}(%conv2d_res)
        return (%res))");

  auto conv_op_fused_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %input_size:int[], %dummy_attr:str):
        %attr: str = prim::Constant[value="${op_attr}"]()
        %packed_weight_bias : __torch__.torch.classes.mkldnn.Conv2dOpContext = mkldnn_prepacked::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %input_size, %attr)
        %res = mkldnn_prepacked::conv2d_run(%input, %packed_weight_bias)
        return (%res))");

  for (const auto& op : fusion_operators) {
    at::jit::TemplateEnv env;
    env.s("op", op);

    at::jit::TemplateEnv env_fused;
    env_fused.s("op_attr", PrepareAttr(op));

    rewriter.RegisterRewritePattern(
        conv_op_rstring.format(env), conv_op_fused_rstring.format(env_fused));
  }

  rewriter.runOnGraph(graph);
}

void PrePackingOpsFolder(Block* b) {
  auto is_foldable_op = [](const Node* n) -> bool {
    return (
        n->kind() ==
        Symbol::fromQualString("mkldnn_prepacked::conv2d_prepack"));
  };

  std::unordered_set<Node*> nodes_to_delete;
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      PrePackingOpsFolder(block);
    }
    if (is_foldable_op(n)) {
      auto optional_outputs = torch::jit::runNodeIfInputsAreConstant(n);
      if (optional_outputs) {
        auto outputs = optional_outputs.value();
        TORCH_CHECK(outputs.size() == 1, "Prepack ops have single output");
        Value* prepack_op_value = n->output(0);
        auto graph = n->owningGraph();
        WithInsertPoint ins(prepack_op_value->node());
        auto weak_class_obj =
            outputs[0].toObject()->copy_to_weak_compilation_ref();
        Value* packed_weight = graph->insertConstant(weak_class_obj)
                                   ->setType(n->output(0)->type());
        prepack_op_value->replaceAllUsesWith(packed_weight);
        nodes_to_delete.insert(n);
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

void FoldPrePackingOps(std::shared_ptr<Graph>& graph) {
  PrePackingOpsFolder(graph->block());
}

void FuseMkldnn(std::shared_ptr<Graph>& graph) {
  insertMkldnnPrePackedOps(graph);
  GRAPH_DEBUG(
      "After insertMkldnnPrePackedOps, before FuseReluWithPackedOps\n", *graph);
  FuseReluWithPackedOps(graph);
  GRAPH_DEBUG(
      "After FuseReluWithPackedOps, before FoldPrePackingOps\n", *graph);
  FoldPrePackingOps(graph);
}

#else

void FuseMkldnn(std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("MKLDNN Not enabled", graph);
}

#endif // AT_MKLDNN_ENABLED()

} // namespace jit
} // namespace torch
