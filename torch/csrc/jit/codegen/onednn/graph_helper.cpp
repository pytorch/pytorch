#include <ATen/native/mkldnn/LlgaTensorImpl.h>
#include <torch/csrc/jit/codegen/onednn/graph_helper.h>

#include <ATen/core/functional.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

using opkind = dnnl::graph::op::kind;

void fixConvOptionalBias(Node* node) {
  if (!node->input(2)->mustNotBeNone()) {
    // Replace non-existent optional bias with const None
    auto g = node->owningGraph();
    auto n = g->createNone();
    auto v = n->insertBefore(node)->output();
    node->replaceInput(2, v);
  }
}

c10::optional<size_t> getDimensions(Value* v) {
  if (v->type()->isSubtypeOf(TensorType::get()))
    return v->type()->cast<TensorType>()->sizes().size();
  else
    return c10::nullopt;
}

Operator makeWildcardOp(Node* node) {
  auto o = Operator(node, opkind::Wildcard);
  // wildcard op contains only topology info
  for (size_t i = 0; i < node->inputs().size(); i++)
    o.setInput(i);
  for (size_t i = 0; i < node->outputs().size(); i++)
    o.setOutput(i);
  return o;
}

#define REQ(cond)                                     \
  if (!(cond)) {                                      \
    GRAPH_DEBUG("Unsupported condition " #cond "\n"); \
    return makeWildcardOp(node);                      \
  }

Operator makeEltwiseOp(Node* node, opkind kind) {
  return Operator(node, kind).setInput(0).setOutput(0);
}

Operator makeBinaryOp(Node* node, opkind kind) {
  REQ(node->input(0)->type()->isSubtypeOf(TensorType::get()) &&
      node->input(1)->type()->isSubtypeOf(TensorType::get()))
  return Operator(node, kind).setInput(0, 1).setOutput(0);
}

Operator createOperator(Node* node) {
  switch (node->kind()) {
    case aten::conv2d: {
      fixConvOptionalBias(node);
      return Operator(node, opkind::Convolution)
          .setInput(0, 1, 2)
          .setOutput(0)
          .setAttr("strides", Operator::Ints, 3)
          .setAttr("pads_begin", Operator::Ints, 4)
          .setAttr("pads_end", Operator::Ints, 4)
          .setAttr("dilations", Operator::Ints, 5)
          .setAttr("groups", Operator::Int, 6)
          .setAttr("filter_format", std::string("OIX"));
    }

    case aten::_convolution: {
      bool transposed = Operator::Bool(node, 6);
      REQ(!transposed);

      return Operator(node, opkind::Convolution)
          .setInput(0, 1, 2)
          .setOutput(0)
          .setAttr("strides", Operator::Ints, 3)
          .setAttr("pads_begin", Operator::Ints, 4)
          .setAttr("pads_end", Operator::Ints, 4)
          .setAttr("dilations", Operator::Ints, 5)
          .setAttr("groups", Operator::Int, 8)
          .setAttr("filter_format", std::string("OIX"));
    }

    case aten::batch_norm: {
      auto training = toIValue(node->input(5));
      REQ(training.has_value()); // cannot get training status in script mode
      REQ(!training->toBool()); // TODO: support bn training
      return Operator(node, opkind::BatchNormInference)
          .setInput(0, 1, 2, 3, 4)
          .setOutput(0)
          .setAttr("epsilon", Operator::Float, 7);
    }

    case aten::layer_norm: {
      auto normalized_shape = Operator::Ints(node, 1);
      REQ(normalized_shape.size() == 1);
      return Operator(node, opkind::LayerNorm)
          .setInput(0, 2, 3)
          .setOutput(0)
          .setAttr("epsilon", Operator::Float, 4)
          .setAttr("keep_stats", false);
    }

    case aten::add:
      return makeBinaryOp(node, opkind::Add);

    case aten::mul:
      return makeBinaryOp(node, opkind::Multiply);

    case aten::tanh:
      return makeEltwiseOp(node, opkind::Tanh);

    case aten::relu:
      return makeEltwiseOp(node, opkind::ReLU);

    case aten::elu:
      return makeEltwiseOp(node, opkind::Elu)
          .setAttr("alpha", Operator::Float, 1);

    case aten::sigmoid:
      return makeEltwiseOp(node, opkind::Sigmoid);

    case aten::gelu:
      return makeEltwiseOp(node, opkind::GELU);

    case aten::sqrt:
      return makeEltwiseOp(node, opkind::Sqrt);

    case aten::abs:
      return makeEltwiseOp(node, opkind::Abs);

    case aten::square:
      return makeEltwiseOp(node, opkind::Square);

    case aten::hardtanh:
      return makeEltwiseOp(node, opkind::HardTanh)
          .setAttr("min", Operator::Float, 1)
          .setAttr("max", Operator::Float, 2);

    case aten::relu6:
      return makeEltwiseOp(node, opkind::HardTanh)
          .setAttr("min", 0.f)
          .setAttr("max", 6.f);

    case aten::softmax: {
      auto axis = Operator::Int(node, 1);
      return Operator(node, opkind::SoftMax)
          .setInput(0)
          .setOutput(0)
          .setAttr("axis", axis);
    }

    case aten::cat: {
      auto o = Operator(node, opkind::Concat);
      REQ(node->input(0)->node()->kind() == prim::ListConstruct);
      REQ(node->input(0)->uses().size() == 1);
      REQ(node->input(1)->node()->kind() == prim::Constant);
      // aten::cat needs a special handling since it takes a Tensor[] as input.
      // We set the inputs of ListConstruct as the inputs of cat.
      //
      // Pytorch IR:                              LLGA sees:
      //     %a    %b     %c          %dim              %a    %b    %c
      //      \     |     /             |                \     |    /
      //   prim::ListConstruct   prim::Constant     llga::Concat[axis=%dim]
      //                    \      /
      //                    aten::cat
      auto listConstruct = node->input(0)->node();
      for (auto input : listConstruct->inputs())
        o.setInputValue(input);
      return o.setOutput(0).setAttr("axis", Operator::Int, 1);
    }

    case aten::max_pool2d: {
      REQ(node->input(1)->node()->kind() == prim::Constant);

      auto rounding_type = Operator::Bool(node, 5) ? "ceil" : "floor";
      return Operator(node, opkind::MaxPool)
          .setInput(0)
          .setOutput(0)
          .setAttr("kernel", Operator::Ints, 1)
          .setAttr("strides", Operator::Ints, 2)
          .setAttr("pads_begin", Operator::Ints, 3)
          .setAttr("pads_end", Operator::Ints, 3)
          .setAttr("dilations", Operator::Ints, 4)
          .setAttr("rounding_type", std::string(rounding_type));
    }

    case aten::avg_pool2d: {
      // TODO: do we need add check for all Constant?
      REQ(node->input(1)->node()->kind() == prim::Constant);

      auto rounding_type = Operator::Bool(node, 4) ? "ceil" : "floor";
      auto divisor_override = toIValue(node->input(6));
      REQ(divisor_override->isNone());
      return Operator(node, opkind::AvgPool)
          .setInput(0)
          .setOutput(0)
          .setAttr("kernel", Operator::Ints, 1)
          .setAttr("strides", Operator::Ints, 2)
          .setAttr("pads_begin", Operator::Ints, 3)
          .setAttr("pads_end", Operator::Ints, 3)
          .setAttr("exclude_pad", !Operator::Bool(node, 5))
          .setAttr("rounding_type", std::string(rounding_type));
    }

    case aten::matmul: {
      auto dim0 = getDimensions(node->input(0)).value_or(-1);
      auto dim1 = getDimensions(node->input(1)).value_or(-1);
      // TODO: support all shape combinations
      REQ((dim0 == 2 && dim1 == 2) || (dim0 == 4 && dim1 == 4) ||
          (dim0 == 3 && dim1 == 2));
    } // fall through
    case aten::mm: {
      return Operator(node, opkind::MatMul).setInput(0, 1).setOutput(0);
    }

    case aten::linear: {
      auto dim0 = getDimensions(node->input(0)).value_or(-1);
      auto dim1 = getDimensions(node->input(1)).value_or(-1);
      // REQ(dim1 == 2);

      return Operator(node, opkind::MatMul)
          .setInput(0, 1, 2)
          .setOutput(0)
          .setAttr("transpose_b", true);
    }

    default:
      return makeWildcardOp(node);
  }
}

dnnl::graph::op createLlgaOp(Node* node) {
  return createOperator(node).llgaOp();
}

bool isSupported(Node* node) {
  return createOperator(node).kind() != opkind::Wildcard;
};

DeviceType inferDeviceFromValue(Value* v) {
  auto tt = v->type()->cast<TensorType>();
  if (!tt)
    return at::kCPU;
  auto device = tt->device();
  if (!device)
    return at::kCPU;
  return device->type();
}

DeviceType inferDevice(const std::shared_ptr<Graph>& graph) {
  auto dt = inferDeviceFromValue(graph->inputs()[0]);
  TORCH_CHECK(
      std::all_of(
          graph->inputs().begin(),
          graph->inputs().end(),
          [dt](Value* v) { return inferDeviceFromValue(v) == dt; }),
      "All inputs must have the same deive type");
  return dt;
}

dnnl::graph::engine::kind getLlgaEngineKind(DeviceType type) {
  switch (type) {
    case DeviceType::CPU:
      return dnnl::graph::engine::kind::cpu;
    default:
      TORCH_CHECK(false, "Not support device type ", type);
  }
}

void mayAddListConstructIntoConcatPartition(
    Node* n,
    OpPartitionMap& opToOwningPartition) {
  // Since prim::ListConstruct is not visible to the LLGA,
  // it will not be in any partition returned from partfuseritioning results.
  // We need rewrite opToOwningPartition to make the prim::ListConstruct to be
  // virtually in the same partition with the aten::cat, so that
  // prim::ListConstruct can be fused into the fusion group by graph fuser
  if (n->kind() == aten::cat && opToOwningPartition.has(n)) {
    auto listConstrcut = n->input(0)->node();
    auto partitionId = opToOwningPartition.get(n);
    opToOwningPartition.add(listConstrcut, partitionId);
  }
}

LlgaGraphHelper::LlgaGraphHelper(
    const std::shared_ptr<Graph>& graph,
    dnnl::graph::partition::policy policy) {
  auto deviceType = inferDevice(graph);
  auto engineKind = getLlgaEngineKind(deviceType);
  dnnl::graph::graph g{engineKind};

  GRAPH_DEBUG("Constructing LLGA graph");
  // TODO: select nodes in top-level block for now
  for (auto* node : graph->block()->nodes()) {
    auto op = createLlgaOp(node);
    g.add_op(op);
    GRAPH_DEBUG("  Added node ", node->kind().toQualString());
  }

  GRAPH_DEBUG("Get Partitions");
  std::vector<dnnl::graph::partition> partitions = g.get_partitions(policy);
  // excluded unsupported Wildcard partitions
  for (size_t partId = 0; partId < partitions.size(); partId++) {
    if (partitions[partId].is_supported())
      partitions_.push_back(partitions[partId]);
  }

  GRAPH_DEBUG("  Got #partitions: ", partitions_.size());
  for (size_t partId = 0; partId < partitions_.size(); partId++) {
    for (auto opId : partitions_[partId].get_ops()) {
      opToOwningPartition_.add(opId, partId);
    }
  }

  // Scanning the graph again for post processing
  for (auto* node : graph->block()->nodes()) {
    mayAddListConstructIntoConcatPartition(node, opToOwningPartition_);
  }
}

bool LlgaGraphHelper::isLlgaSubgraph(const Node* node) {
  return node->hasAttribute(attr::Subgraph) &&
      node->kind() == prim::LlgaFusionGroup;
}

bool LlgaGraphHelper::shouldMerge(Node* toMerge, Node* subgraph) {
  TORCH_CHECK(
      isLlgaSubgraph(subgraph),
      "The consumer node does not contain a subgraph");
  if (!shouldConsiderForMerge(toMerge)) {
    return false;
  }
  return opToOwningPartition_.get(toMerge) ==
      opToOwningPartition_.get(subgraph);
}

bool isViewOp(Node* n) {
  switch (n->kind()) {
    case aten::view:
    case aten::view_as:
    case aten::reshape:
    case aten::reshape_as:
    case aten::transpose:
    case aten::expand:
    case aten::expand_as:
      return true;
  }
  return false;
}

bool LlgaGraphHelper::shouldConsiderForMerge(Node* node) {
  // if we're already in the process of merging
  if (isLlgaSubgraph(node)) {
    return true;
  }
  if (isViewOp(node)) {
    return false;
  }
  return opToOwningPartition_.has(node);
}

Node* LlgaGraphHelper::createSingletonSubgraph(Node* n, AliasDb& aliasDb) {
  auto partitionId = opToOwningPartition_.get(n);
  GRAPH_DEBUG(
      "Creating FusionGroup_", partitionId, " for ", n->kind().toQualString());
  auto group = SubgraphUtils::createSingletonSubgraphAndUpdateAliasing(
      n, prim::LlgaFusionGroup, aliasDb);
  opToOwningPartition_.add(group, partitionId);
  LlgaNodeWrapper(group).initOutputLayouts();
  return group;
}

void LlgaGraphHelper::mergeNodeIntoSubgraph(
    Node* toMerge,
    Node* subgraphNode,
    AliasDb& aliasDb) {
  if (isLlgaSubgraph(toMerge)) {
    GRAPH_DEBUG(
        "Merging ",
        toMerge->kind().toQualString(),
        "_",
        opToOwningPartition_.get(toMerge),
        " into ",
        subgraphNode->kind().toQualString(),
        "_",
        opToOwningPartition_.get(subgraphNode));
  } else {
    GRAPH_DEBUG(
        "Merging ",
        toMerge->kind().toQualString(),
        " into ",
        subgraphNode->kind().toQualString(),
        "_",
        opToOwningPartition_.get(subgraphNode));
  }

  SubgraphUtils::mergeNodeIntoSubgraphAndUpdateAliasing(
      toMerge, subgraphNode, aliasDb);
}

void LlgaGraphHelper::unmergeIfAnyNodeIsMissing(Node* subgraphNode) {
  TORCH_CHECK(isLlgaSubgraph(subgraphNode), "Cannot unmerge a non-LLGA node");

  auto partitionId = opToOwningPartition_.get(subgraphNode);
  auto expectOpNum = partitions_[partitionId].get_ops_num();
  auto actualOpNum = countSupportedOps(subgraphNode->g(attr::Subgraph));

  if (expectOpNum != actualOpNum) {
    GRAPH_DEBUG(
        "Unmerging FusionGroup_",
        partitionId,
        ". Expected ",
        expectOpNum,
        " ops, but got ",
        actualOpNum,
        " ops.");
    SubgraphUtils::unmergeSubgraph(subgraphNode);
  }
}

size_t LlgaGraphHelper::countSupportedOps(
    const std::shared_ptr<Graph>& graph) const {
  // TODO: count nodes in top-level block for now
  size_t cnt = 0;
  for (auto* node : graph->block()->nodes())
    if (isSupported(node))
      cnt++;
  return cnt;
}

std::vector<dnnl::graph::partition> LlgaGraphHelper::getPartitions() const {
  return partitions_;
}

LlgaNodeWrapper::LlgaNodeWrapper(const Node* node)
    : n(const_cast<Node*>(node)) {
  TORCH_CHECK(
      LlgaGraphHelper::isLlgaSubgraph(n), "Cannot wrap a non-LLGA fusion node");
}

void LlgaNodeWrapper::setOpaqueLayout(size_t offset) {
  TORCH_CHECK(offset < n->outputs().size(), "Invalid output offset ", offset);
  auto& layouts =
      const_cast<std::vector<int64_t>&>(n->is(attr::output_layouts));
  layouts.at(offset) = 1;
}

bool LlgaNodeWrapper::useOpaqueLayout(size_t offset) const {
  TORCH_CHECK(offset < n->outputs().size(), "Invalid output offset ", offset);
  return n->is(attr::output_layouts)[offset] == 1;
}

void LlgaNodeWrapper::initOutputLayouts() {
  if (n->hasAttribute(attr::output_layouts)) {
    return;
  }

  // Init all output layouts as undef
  std::vector<int64_t> layouts(n->outputs().size(), 0);
  n->is_(attr::output_layouts, layouts);
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch