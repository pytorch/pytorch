#include <ATen/Utils.h>

#include <ATen/Config.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/frozen_conv_folding.h>
#include <torch/csrc/jit/passes/frozen_ops_to_mkldnn.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
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

Operation BroadOp(const Node* node) {
  return [](Stack* stack) {
    auto b = pop(stack).toTensor();
    auto a = pop(stack).toTensor();
    auto b_size = b.sizes();
    auto a_size = a.sizes();
    if (a_size.equals(b_size)) {
      push(stack, a, b);
    } else {
      auto out_size = at::infer_size(a_size, b_size);
      if (a_size.equals(out_size)) {
        push(stack, a);
      } else {
        push(stack, a.to_dense().expand(out_size).to_mkldnn());
      }
      if (b_size.equals(out_size)) {
        push(stack, b);
      } else {
        push(stack, b.to_dense().expand(out_size).to_mkldnn());
      }
    }
  };
}

const RegisterOperators BroadOpReg({
    torch::jit::Operator(
        prim::BroadcastMKLDNNTensors,
        BroadOp,
        AliasAnalysisKind::INTERNAL_SPECIAL_CASE),
});

Operation ConstantMKLDNNTensorOp(const Node* node) {
  const auto& t = node->t(attr::value);
  return [t](Stack* stack) {
    push(stack, t);
    return 0;
  };
}

// This is registered as its own op instead of as prim::Constant bc it does not
// serialize which is an invariant of prim::Constant
// TODO: make mkldnn tensor serialize...
const RegisterOperators MKLDNNConstantOp({
    torch::jit::Operator(
        prim::ConstantMKLDNNTensor,
        ConstantMKLDNNTensorOp,
        AliasAnalysisKind::INTERNAL_SPECIAL_CASE),
});

Node* createConstantMKLDNNTensorOp(Graph* g, const Tensor& mkldnn_tensor) {
  TORCH_INTERNAL_ASSERT(mkldnn_tensor.is_mkldnn());
  auto op = g->create(prim::ConstantMKLDNNTensor);
  op->t_(attr::value, mkldnn_tensor);
  return op;
}

bool supportedMKLDNNWeight(const Tensor& weight) {
  return weight.device().is_cpu() && weight.dtype() == c10::ScalarType::Float &&
      weight.ndimension() != 0;
}

void replaceInputWithMKLDNNTensor(Node* n, size_t index) {
  Value* input = n->inputs().at(index);
  auto mkldnn_tensor = constant_as<Tensor>(input)->to_mkldnn();
  auto mkldnn_tensor_value =
      createConstantMKLDNNTensorOp(n->owningGraph(), mkldnn_tensor)
          ->insertBefore(n)
          ->output();
  mkldnn_tensor_value->setDebugName(input->debugName() + "_mkldnn");
  n->replaceInputWith(input, mkldnn_tensor_value);
}

void replaceInputWithMKLDNNTensor(
    Node* n,
    const std::string& name,
    const at::Tensor& mkldnn_tensor) {
  Value* input = n->namedInput(name);
  auto mkldnn_tensor_value =
      createConstantMKLDNNTensorOp(n->owningGraph(), mkldnn_tensor)
          ->insertBefore(n)
          ->output();
  mkldnn_tensor_value->setDebugName(input->debugName() + "_mkldnn");
  n->replaceInputWith(input, mkldnn_tensor_value);
}

void replaceInputWithMKLDNNTensor(Node* n, const std::string& name) {
  Value* input = n->namedInput(name);
  auto mkldnn_tensor = constant_as<Tensor>(input)->to_mkldnn();
  replaceInputWithMKLDNNTensor(n, name, mkldnn_tensor);
}

void moveConvWeightsToMKLDNN(Node* conv) {
  auto conv_w_mkldnn =
      constant_as<Tensor>(conv->namedInput("weight")).value().to_mkldnn();
  std::vector<int64_t> padding =
      toIValue(conv->namedInput("padding"))->toIntVector();
  std::vector<int64_t> stride =
      toIValue(conv->namedInput("stride"))->toIntVector();
  std::vector<int64_t> dilation =
      toIValue(conv->namedInput("dilation"))->toIntVector();
  auto groups = constant_as<int64_t>(conv->namedInput("groups")).value();

  if (conv->kind() == aten::conv2d) {
    conv_w_mkldnn = mkldnn_reorder_conv2d_weight(
        conv_w_mkldnn, padding, stride, dilation, groups);
  } else if (conv->kind() == aten::conv3d) {
    conv_w_mkldnn = mkldnn_reorder_conv3d_weight(
        conv_w_mkldnn, padding, stride, dilation, groups);
  } else {
    TORCH_INTERNAL_ASSERT(false);
  }
  replaceInputWithMKLDNNTensor(conv, "weight", conv_w_mkldnn);

  if (conv->namedInput("bias")->type() != NoneType::get()) {
    replaceInputWithMKLDNNTensor(conv, "bias");
  }
}

void moveWeightsToMKLDNN(Node* n) {
  // conv goes through special pathway so we can call mkldnn reorder conv
  // primitive
  if (n->kind() == aten::conv2d || n->kind() == aten::conv3d) {
    moveConvWeightsToMKLDNN(n);
  } else {
    for (size_t i = 0; i < n->inputs().size(); ++i) {
      if (!n->input(i)->type()->cast<TensorType>() ||
          n->input(i)->node()->kind() != prim::Constant) {
        continue;
      }
      replaceInputWithMKLDNNTensor(n, i);
    }
  }
}

void computeSubgraphInMKLDNN(Node* subgraph_node) {
  auto graph = subgraph_node->owningGraph();
  Value* none_value = nullptr;
  {
    WithInsertPoint guard(subgraph_node);
    none_value = graph->insertConstant(IValue());
  }
  for (size_t i = 0; i < subgraph_node->inputs().size(); ++i) {
    Value* v = subgraph_node->inputs().at(i);
    if (!v->type()->cast<TensorType>()) {
      continue;
    }
    auto to_mkldnn =
        graph->create(c10::Symbol::fromQualString("aten::to_mkldnn"), 1)
            ->insertBefore(subgraph_node);
    to_mkldnn->addInput(v);
    to_mkldnn->addInput(none_value);
    subgraph_node->replaceInput(i, to_mkldnn->output());
  }

  for (size_t i = 0; i < subgraph_node->outputs().size(); ++i) {
    Value* v = subgraph_node->outputs().at(i);
    if (!v->type()->cast<TensorType>()) {
      continue;
    }
    auto from_mkldnn =
        graph
            ->create(
                c10::Symbol::fromQualString("aten::to_dense"), {v, none_value})
            ->insertAfter(subgraph_node);
    v->replaceAllUsesAfterNodeWith(from_mkldnn, from_mkldnn->output());
  }

  auto subgraph = SubgraphUtils::getSubgraph(subgraph_node);
  for (auto it = subgraph->block()->nodes().begin();
       it != subgraph->block()->nodes().end();) {
    Node* body_node = *it;
    it++;

    moveWeightsToMKLDNN(body_node);

    if (body_node->kind() == aten::add || body_node->kind() == aten::mul) {
      auto node = body_node->owningGraph()->create(
          Symbol::prim("BroadcastMKLDNNTensors"),
          {body_node->inputs().at(0), body_node->inputs().at(1)},
          2);
      node->insertBefore(body_node);
      body_node->replaceInput(0, node->outputs().at(0));
      body_node->replaceInput(1, node->outputs().at(1));
    }
  }
}

bool nonConstantParameters(Node* n) {
  for (size_t i = 1; i < n->inputs().size(); i++) {
    if (n->inputs().at(i)->node()->kind() != prim::Constant) {
      return true;
    }
  }
  return false;
}

bool frozenMkldnnCompatibleLinearNode(Node* n) {
  if (nonConstantParameters(n)) {
    return false;
  }

  if (n->kind() != aten::linear) {
    return false;
  }

  auto weight = constant_as<Tensor>(n->namedInput("weight")).value();
  return supportedMKLDNNWeight(weight);
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
  return supportedMKLDNNWeight(weight);
}

// [mkldnn perf strategy]
// Certain ops - aten::linear, aten::conv2d, aten::conv3d - provide a huge speed
// up just by converting the constant weights to MKLDNN AOT, and then at runtime
// converting the non-constant input to_mkldnn before the op, and then back to
// its original layout after the op. The speed up holds even if you end up
// converting the input to_mkldnn and output back to_dense. We start groups of
// ops to compute in MKLDNN only from these ops that are a strict speedup. Then,
// we expand the groups to include operators which are computable in MKLDNN &
// are roughly perf equal to eager. We do this in the hopes of joining multiple
// fast nodes together, saving to_mkldnn and to_dense conversions.
//
// MKLDNN only supports float32 inputs for aten::linear, aten::conv2d &
// aten::conv3d. We only fuse these nodes if the weights are float32, and then
// we only include operators which we can prove will execute in float32. By
// fusing topologically we can maintain the invariant that all tensor types in
// the graph are floating point. In fusing Conv-> Add -> Relu -> Conv we start
// with the first Conv, know that the output is float, and can then safely merge
// Add and Relu. If we started with the last Conv, it would be difficult to
// prove in our first pass that the Add's inputs were both float32 without first
// fusing the first conv.

class MKLDNNSubgraphSlicer {
 public:
  MKLDNNSubgraphSlicer(
      Block* block,
      std::shared_ptr<Graph> graph,
      AliasDb& aliasDb)
      : block_(block), graph_(std::move(graph)), aliasDb_(aliasDb) {}

  void run() {
    // We maintain alias db correctness in-place while building up the autodiff
    // subgraphs, however it is difficult to preserve correctness when
    // un-inlining autodiff subgraphs. We first recursively construct all
    // subgraphs and then unmerge them into the graph
    buildupSubgraphs();
    computeSubgraphsInMKLDNN();
    // Run CSE globally onceto eliminate duplicates that may have occurred
    // while inlining subgraphs.
    EliminateCommonSubexpression(graph_);
  }

  void buildupSubgraphs() {
    // We need to run the slicer multiple times in order to get all merge
    // opportunities. This is because moveBeforeTopologicalValid may reorder
    // nodes to be AFTER the current iteration point. In order to properly
    // consider those nodes for merging, we need run the pass until no changes
    // have been made.
    //
    // Example:
    //   c = f(a, b)
    //   d = f(c)
    //   e = f(d)  <- iter is here, moving upward
    // After c.moveBeforeTopologicallyValid(e), we have:
    //   c = f(a, b)
    //   e = f(d)  <- iter still here
    //   d = f(c)  <- this was node moved on the other side.

    bool any_changed = true;
    while (any_changed) {
      any_changed = false;
      for (auto it = block_->nodes().begin(); it != block_->nodes().end();) {
        bool changed = false;
        std::tie(it, changed) = scanNode(*it);
        any_changed |= changed;
      }
    }

    // Construct Subgraphs Recursively
    for (Node* n : block_->nodes()) {
      for (auto subBlock : n->blocks()) {
        MKLDNNSubgraphSlicer(subBlock, graph_, aliasDb_).buildupSubgraphs();
      }
    }
  }

  static bool MKLDNNGroupStart(Node* node) {
    // if we're already in the process of merging
    if (node->kind() == prim::MKLDNNGroup) {
      return true;
    }
    // see [mkldnn perf strategy]
    return frozenMkldnnCompatibleLinearNode(node) ||
        frozenMkldnnCompatibleConvNode(node);
  }

 private:
  // MKLDNN only supports floats of dimension > 0, so we only support
  // Tensors who have a known type or were previously verified
  // to be usable in an MKLDNN Group
  bool tensorInputIsMKLDNNSupported(Value* v, Node* v_use) {
    auto const_tensor = constant_as<Tensor>(v);
    if (const_tensor) {
      return supportedMKLDNNWeight(*const_tensor);
    }
    auto k = v->node()->kind();
    if (k == prim::MKLDNNGroup || k == prim::ConstantMKLDNNTensor ||
        k == aten::to_mkldnn) {
      return true;
    }
    for (const auto& use : v->uses()) {
      if (use.user->kind() == aten::to_mkldnn &&
          v_use->owningBlock() == use.user->owningBlock()) {
        return true;
      }
    }
    return false;
  }

  // We include ops here which are roughly perf-equivalent in mkldnn as with
  // aten (single & multithreaded) and whose inputs & outputs are float32.
  bool computableInMKLDNN(Node* n) {
    for (Value* v : n->inputs()) {
      if (v->type()->cast<TensorType>() &&
          !(tensorInputIsMKLDNNSupported(v, n))) {
        return false;
      }
    }
    // unary ops we dont need to prove anything else than
    // the input is mkldnn supported
    switch (n->kind()) {
      case aten::relu:
      case aten::sigmoid:
      // TODO: max_pool on mkldnn can be slower than in eager. ideally, we'd
      // only fuse it if we knew including max_pool lead to fewer layout
      // conversions. from initial testing including it speeds up models
      case aten::max_pool2d:
      case aten::max_pool3d:
        return true;
    }

    if (n->kind() == aten::add) {
      // mkldnn doesn't currently support Tensor-Scalar add
      for (size_t i = 0; i < 2; i++) {
        if (!n->inputs().at(i)->type()->cast<TensorType>()) {
          return false;
        }
      }
      return true;
    }
    // TODO: dropout removal. mkldnn doesnt support train=True
    if (n->kind() == aten::dropout) {
      auto train = constant_as<bool>(n->namedInput("train")).value();
      return train == false;
    }
    return false;
  }

  void computeSubgraphsInMKLDNN() {
    auto curNode = *block_->nodes().begin();
    while (curNode != *block_->nodes().end()) {
      auto nextNode = curNode->next();
      if (curNode->kind() == prim::MKLDNNGroup) {
        computeSubgraphInMKLDNN(curNode);
        SubgraphUtils::unmergeSubgraph(curNode);
      }
      curNode = nextNode;
    }
    for (Node* n : block_->nodes()) {
      for (Block* b : n->blocks()) {
        MKLDNNSubgraphSlicer(b, graph_, aliasDb_).computeSubgraphsInMKLDNN();
      }
    }
  }

  bool shouldConsiderForMerge(Node* node) {
    // if we're already in the process of merging
    if (node->kind() == prim::MKLDNNGroup) {
      return true;
    }
    return frozenMkldnnCompatibleLinearNode(node) ||
        frozenMkldnnCompatibleConvNode(node) || computableInMKLDNN(node);
  }

  std::pair<graph_node_list::iterator, bool> scanNode(Node* producer) {
    if (MKLDNNGroupStart(producer)) {
      if (producer->kind() != prim::MKLDNNGroup) {
        producer = SubgraphUtils::createSingletonSubgraphAndUpdateAliasing(
            producer, prim::MKLDNNGroup, aliasDb_);
      }
      std::vector<Node*> output_nodes;
      for (Value* v : producer->outputs()) {
        for (const auto& use : v->uses()) {
          output_nodes.push_back(use.user);
        }
      }
      std::sort(
          output_nodes.begin(), output_nodes.end(), [&](Node* a, Node* b) {
            return a->isBefore(b);
          });
      for (auto output_node : output_nodes) {
        if (auto group = tryMerge(producer, output_node)) {
          // we successfully merged, so the new group's `outputs` may have
          // changed. So rescan the new group for more merging opportunities.
          return std::make_pair(group.value()->iterator()++, true);
        }
      }
    }

    return std::make_pair(++producer->iterator(), false);
  }

  // Try to merge `consumer` into `producer`. If successful, this destroys
  // `consumer` and returns the `producer` group.
  c10::optional<Node*> tryMerge(Node* producer, Node* consumer) {
    AT_ASSERT(producer->kind() == prim::MKLDNNGroup);
    bool canMerge = shouldConsiderForMerge(consumer) &&
        aliasDb_.moveAfterTopologicallyValid(consumer, producer);

    if (!canMerge) {
      return c10::nullopt;
    }

    SubgraphUtils::mergeNodeIntoSubgraphAndUpdateAliasing(
        consumer, producer, aliasDb_);

    return producer;
  }

  Block* block_;
  std::shared_ptr<Graph> graph_;
  AliasDb& aliasDb_;
};

bool containsMKLDNNGroup(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      if (containsMKLDNNGroup(block)) {
        return true;
      }
    }
    if (MKLDNNSubgraphSlicer::MKLDNNGroupStart(n)) {
      return true;
    }
  }
  return false;
}

} // namespace

void ConvertFrozenOpsToMKLDNN(std::shared_ptr<Graph>& graph) {
#if AT_MKLDNN_ENABLED()
  GRAPH_DUMP("Before convert frozen ops to mkldnn", graph);
  // TODO: replace conv1d with conv2d ?
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);
  if (containsMKLDNNGroup(graph->block())) {
    // Only remove tensor mutation if we know we're going to create speedups
    // with mkldnn. Only supporting functional ops simplifies this pass bc
    // running an op in mkldnn removes the aliasing relationships that
    // previously existed between input and output.
    RemoveTensorMutation(graph, [](Node* node_to_functionalize) {
      static std::unordered_set<Symbol> mkldnn_ops = {
          aten::add_,
          aten::mul_,
          aten::relu_,
          aten::dropout_,
          aten::sigmoid_,
      };
      return mkldnn_ops.count(node_to_functionalize->kind()) != 0;
    });
    AliasDb db(graph);
    MKLDNNSubgraphSlicer(graph->block(), graph, db).run();
    EliminateDeadCode(graph);
    GRAPH_DUMP("After convert frozen ops to mkldnn", graph);
  } else {
    GRAPH_DUMP("No mkldnn compatible frozen nodes", graph);
  }
#else
  GRAPH_DUMP("MKLDNN Not enabled", graph);
#endif
}

} // namespace jit
} // namespace torch
