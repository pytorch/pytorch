#include <torch/csrc/jit/codegen/onednn/LlgaTensorImpl.h>
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
  if (node->namedInput("bias")->mustNotBeNone() == false) {
    // Replace non-existent optional bias with const None
    auto g = node->owningGraph();
    auto n = g->createNone();
    auto v = n->insertBefore(node)->output();
    node->replaceInput(2, v);
  }
}

c10::optional<size_t> getDimensions(Value* v) {
  if (v->type()->isSubtypeOf(TensorType::get())) {
    return v->type()->cast<TensorType>()->sizes().size();
  } else {
    return c10::nullopt;
  }
}

// PyTorch ops that can't otherwise be mapped to oneDNN Graph ops are mapped as
// Wildcards instead. They make the integration code with PyTorch simpler by
// passing every op to the oneDNN Graph library in the add_op call -
// no need to check beforehand whether the op is supported by oneDNN Graph or
// not oneDNN Graph ops separated by wildcards don't end up in the same
// partition.
Operator makeWildcardOp(Node* node) {
  auto o = Operator(node, opkind::Wildcard);
  // wildcard op contains only topology info
  for (size_t i = 0; i < node->inputs().size(); i++) {
    o.setInput(i);
  }
  for (size_t i = 0; i < node->outputs().size(); i++) {
    o.setOutput(i);
  }
  return o;
}

// If we don't meet a certain condition to map a PyTorch op to a oneDNN Graph
// op, then we create a wildcard op corresponding to that PyTorch op instead.
#define REQUIRE(cond)                                 \
  if (!(cond)) {                                      \
    GRAPH_DEBUG("Unsupported condition " #cond "\n"); \
    return makeWildcardOp(node);                      \
  }

Operator makeEltwiseOp(Node* node, opkind kind) {
  return Operator(node, kind).setInput(0).setOutput(0);
}

Operator makeBinaryOp(Node* node, opkind kind) {
  REQUIRE(
      node->input(0)->type()->isSubtypeOf(TensorType::get()) &&
      node->input(1)->type()->isSubtypeOf(TensorType::get()))
  return Operator(node, kind).setInput(0, 1).setOutput(0);
}

// Map a PyTorch op to its corresponding oneDNN Graph op.
// If mapping isn't possible, then create a wildcard op instead.
// The mapping is done as per oneDNN Graph op schema defined in
// third_party/ideep/mkl-dnn/src/interface/op_def.hpp.
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
          .setAttr("filter_format", std::string("OIX"))
          .setAttr("data_format", std::string("NCX"));
    }

    case aten::_convolution: {
      bool transposed = toIValue(node->namedInput("transposed"))->toBool();
      REQUIRE(!transposed);

      return Operator(node, opkind::Convolution)
          .setInput(0, 1, 2)
          .setOutput(0)
          .setAttr("strides", Operator::Ints, 3)
          .setAttr("pads_begin", Operator::Ints, 4)
          .setAttr("pads_end", Operator::Ints, 4)
          .setAttr("dilations", Operator::Ints, 5)
          .setAttr("groups", Operator::Int, 8)
          .setAttr("filter_format", std::string("OIX"))
          .setAttr("data_format", std::string("NCX"));
    }

    case aten::batch_norm: {
      auto training = toIValue(node->namedInput("training"));
      REQUIRE(
          training.has_value()); // cannot get training status in script mode
      REQUIRE(!training->toBool()); // TODO: support bn training
      return Operator(node, opkind::BatchNormInference)
          .setInput(0, 1, 2, 3, 4)
          .setOutput(0)
          .setAttr("epsilon", Operator::Float, 7)
          .setAttr("data_format", std::string("NCX"));
    }

    case aten::layer_norm: {
      auto normalized_shape = toIValue(node->namedInput("normalized_shape"));
      REQUIRE(normalized_shape->toIntList().size() == 1);
      return Operator(node, opkind::LayerNorm)
          .setInput(0, 2, 3)
          .setOutput(0)
          .setAttr("epsilon", Operator::Float, 4)
          .setAttr("keep_stats", false);
    }

    case aten::addmm: {
      auto alpha = toIValue(node->namedInput("alpha"));
      auto beta = toIValue(node->namedInput("beta"));
      REQUIRE(
          alpha.has_value() && beta.has_value() && (alpha->toDouble() == 1.0) &&
          (beta->toDouble() == 1.0));
      return Operator(node, opkind::MatMul).setInput(1, 2, 0).setOutput(0);
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
      auto axis = toIValue(node->namedInput("dim"))->toInt();
      return Operator(node, opkind::SoftMax)
          .setInput(0)
          .setOutput(0)
          .setAttr("axis", axis);
    }

    case aten::cat: {
      auto o = Operator(node, opkind::Concat);
      REQUIRE(
          node->namedInput("tensors")->node()->kind() == prim::ListConstruct);
      REQUIRE(node->namedInput("tensors")->uses().size() == 1);
      REQUIRE(node->namedInput("dim")->node()->kind() == prim::Constant);
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
      REQUIRE(
          node->namedInput("kernel_size")->node()->kind() == prim::Constant);

      auto rounding_type =
          toIValue(node->namedInput("ceil_mode"))->toBool() ? "ceil" : "floor";
      return Operator(node, opkind::MaxPool)
          .setInput(0)
          .setOutput(0)
          .setAttr("kernel", Operator::Ints, 1)
          .setAttr("strides", Operator::Ints, 2)
          .setAttr("pads_begin", Operator::Ints, 3)
          .setAttr("pads_end", Operator::Ints, 3)
          .setAttr("dilations", Operator::Ints, 4)
          .setAttr("rounding_type", std::string(rounding_type))
          .setAttr("data_format", std::string("NCX"));
    }

    case aten::avg_pool2d: {
      // TODO: do we need add checks for all Constants?
      REQUIRE(
          node->namedInput("kernel_size")->node()->kind() == prim::Constant);
      auto rounding_type =
          toIValue(node->namedInput("ceil_mode"))->toBool() ? "ceil" : "floor";
      auto divisor_override = toIValue(node->namedInput("divisor_override"));
      REQUIRE(divisor_override->isNone());
      return Operator(node, opkind::AvgPool)
          .setInput(0)
          .setOutput(0)
          .setAttr("kernel", Operator::Ints, 1)
          .setAttr("strides", Operator::Ints, 2)
          .setAttr("pads_begin", Operator::Ints, 3)
          .setAttr("pads_end", Operator::Ints, 3)
          .setAttr("exclude_pad", !Operator::Bool(node, 5))
          .setAttr("rounding_type", std::string(rounding_type))
          .setAttr("data_format", std::string("NCX"));
    }

    case aten::matmul: {
      auto dim0 = getDimensions(node->namedInput("self")).value_or(-1);
      auto dim1 = getDimensions(node->namedInput("other")).value_or(-1);
      // TODO: support all shape combinations
      REQUIRE(
          (dim0 == 2 && dim1 == 2) || (dim0 == 4 && dim1 == 4) ||
          (dim0 == 3 && dim1 == 2));
    } // fall through
    case aten::mm: {
      return Operator(node, opkind::MatMul).setInput(0, 1).setOutput(0);
    }

    case aten::linear: {
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
  if (!tt) {
    return at::kCPU;
  }
  auto device = tt->device();
  if (!device) {
    return at::kCPU;
  }
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
  // 'virtually' in the same partition with the aten::cat, so that
  // prim::ListConstruct can be fused into the fusion group by graph fuser.
  // We emphasize on 'virtually' because get_num_ops() for cat's partition
  // would still return 1.
  if (n->kind() == aten::cat && opToOwningPartition.has(n)) {
    auto listConstrcut = n->namedInput("tensors")->node();
    auto partitionId = opToOwningPartition.get(n);
    opToOwningPartition.add(listConstrcut, partitionId);
  }
}

// Verify that input tensors are compatible with oneDNN Graph.
// Scalars would be converted to 1-D tensors later anyway,
// but they shouldn't be complex-double
// If this check fails, convert op to wildcard
bool checkInputCompatibility(Node* node) {
  auto allInputs = node->inputs();
  for (auto input : allInputs) {
    c10::IValue inputIValue = toIValue(input);
    if (inputIValue.isTensor()) {
      const at::Tensor& tensor = inputIValue.toTensor();
      if (tensor.device() != at::kCPU) {
        return false;
      }
      auto dtype = tensor.scalar_type();
      if ((dtype != at::ScalarType::Float) && (dtype != at::ScalarType::Long)) {
        return false;
      }
    } else if (inputIValue.isScalar()) {
      if (inputIValue.isComplexDouble()) {
        return false;
      }
    }
  }
  return true;
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
    auto kindOfNode = node->kind();
    if (checkInputCompatibility(node)) {
      g.add_op(op);
      GRAPH_DEBUG("  Added node ", kindOfNode.toQualString());
    } else {
      GRAPH_DEBUG("The backend failed to add node ", kindOfNode.toQualString());
      g.add_op(makeWildcardOp(node).llgaOp());
    }

    for (Value* input : node->inputs()) {
      tensorIdToValue_.emplace(input->unique(), input);
    }
  }

  GRAPH_DEBUG("Get Partitions");
  std::vector<dnnl::graph::partition> partitions = g.get_partitions(policy);
  // excluded unsupported Wildcard partitions
  for (auto& partition : partitions) {
    if (partition.is_supported()) {
      partitions_.push_back(partition);
    }
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
      node->kind() == prim::oneDNNFusionGroup;
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

// Except for conv & GEMMs, which should always be handled by oneDNN Graph,
// only use single-op partitions for ops unsupported by NNC, or ops
// that oneDNN executes faster. prim::ListConstruct is an exception, since
// we simply want to fuse it with cat.
bool isBetterSuitedForLLGA(NodeKind kindOfOp) {
  return (
      (kindOfOp == aten::layer_norm) || (kindOfOp == aten::avg_pool2d) ||
      (kindOfOp == aten::matmul) || (kindOfOp == aten::max_pool2d) ||
      (kindOfOp == aten::conv2d) || (kindOfOp == aten::_convolution) ||
      (kindOfOp == aten::mm) || (kindOfOp == aten::linear) ||
      (kindOfOp == aten::cat) || (kindOfOp == prim::ListConstruct));
}

bool LlgaGraphHelper::checkForSingleOpPartition(Node* node) {
  if (opToOwningPartition_.has(node)) {
    auto partitionId = opToOwningPartition_.get(node);
    if (partitions_[partitionId].get_ops_num() == 1) {
      auto kindOfNode = node->kind();
      return isBetterSuitedForLLGA(kindOfNode);
    } else {
      // multi-op partition
      return true;
    }
  } else {
    // this op isn't present in any partition
    return false;
  }
}

bool LlgaGraphHelper::shouldConsiderForMerge(Node* node) {
  // if we're already in the process of merging
  if (isLlgaSubgraph(node)) {
    return true;
  }
  return checkForSingleOpPartition(node);
}

Node* LlgaGraphHelper::createSingletonSubgraph(Node* n, AliasDb& aliasDb) {
  auto partitionId = opToOwningPartition_.get(n);
  GRAPH_DEBUG(
      "Creating FusionGroup_", partitionId, " for ", n->kind().toQualString());
  auto group = SubgraphUtils::createSingletonSubgraphAndUpdateAliasing(
      n, prim::oneDNNFusionGroup, aliasDb);
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
  for (auto* node : graph->block()->nodes()) {
    auto nodeKind = node->kind();
    if ((nodeKind != prim::Constant) && (nodeKind != prim::ListConstruct)) {
      cnt++;
    }
  }
  return cnt;
}

std::vector<dnnl::graph::partition> LlgaGraphHelper::getPartitions() const {
  return partitions_;
}

std::map<size_t, Value*> LlgaGraphHelper::getTensorIdToValue() const {
  return tensorIdToValue_;
}

LlgaNodeWrapper::LlgaNodeWrapper(const Node* node)
    : n(const_cast<Node*>(node)) { // NOLINT
  TORCH_CHECK(
      LlgaGraphHelper::isLlgaSubgraph(n), "Cannot wrap a non-LLGA fusion node");
}

void LlgaNodeWrapper::setOpaqueLayout(size_t offset) {
  TORCH_CHECK(offset < n->outputs().size(), "Invalid output offset ", offset);
  auto& layouts =
      const_cast<std::vector<int64_t>&>(n->is(attr::output_layouts)); // NOLINT
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
