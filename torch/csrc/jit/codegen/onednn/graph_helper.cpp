#include <torch/csrc/jit/codegen/onednn/graph_helper.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace torch::jit::fuser::onednn {

using opkind = dnnl::graph::op::kind;

static void fixConvOptionalBias(Node* node) {
  if (node->namedInput("bias")->mustNotBeNone() == false) {
    // Replace non-existent optional bias with const None
    auto g = node->owningGraph();
    auto n = g->createNone();
    auto v = n->insertBefore(node)->output();
    node->replaceInput(2, v);
  }
}

static std::optional<size_t> getDimensions(Value* v) {
  if (v->type()->isSubtypeOf(TensorType::get())) {
    return v->type()->cast<TensorType>()->sizes().size();
  } else {
    return std::nullopt;
  }
}

// PyTorch ops that can't otherwise be mapped to oneDNN Graph ops are mapped as
// Wildcards instead. They make the integration code with PyTorch simpler by
// passing every op to the oneDNN Graph library in the add_op call -
// no need to check beforehand whether the op is supported by oneDNN Graph or
// not oneDNN Graph ops separated by wildcards don't end up in the same
// partition.
static Operator makeWildcardOp(Node* node) {
  auto o = Operator(node, opkind::Wildcard);
  // wildcard op contains only topology info
  for (size_t i = 0; i < node->inputs().size(); i++) {
    o.setInput(0, i);
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

Operator LlgaGraphHelper::makeEltwiseOp(Node* node, opkind kind) {
  return Operator(node, kind).setInput(0).setOutput(dnnl_graph_, 0);
}

Operator LlgaGraphHelper::makeBinaryOp(Node* node, opkind kind) {
  REQUIRE(
      node->input(0)->type()->isSubtypeOf(TensorType::get()) &&
      node->input(1)->type()->isSubtypeOf(TensorType::get()))
  return Operator(node, kind).setInput(0, 1).setOutput(dnnl_graph_, 0);
}

// Map a PyTorch op to its corresponding oneDNN Graph op.
// If mapping isn't possible, then create a wildcard op instead.
// The mapping is done as per oneDNN Graph op schema defined in
// third_party/ideep/mkl-dnn/src/interface/op_def.hpp.
Operator LlgaGraphHelper::createOperator(Node* node) {
  auto nodeKind = node->kind();
  // we're using an if-else clause instead of a switch statement
  // because we would soon be adding custom ops with function schemas.
  // We would have to use Symbol::fromQualString at that time anyway,
  // but we are okay with this choice, since this code is not in the hot-path.
  if (nodeKind == Symbol::fromQualString("aten::conv2d")) {
    fixConvOptionalBias(node);
    return Operator(node, opkind::Convolution)
        .setInput(0, 1, 2)
        .setOutput(dnnl_graph_, 0)
        .setAttr(dnnl::graph::op::attr::strides, Operator::Ints, 3)
        .setAttr(dnnl::graph::op::attr::pads_begin, Operator::Ints, 4)
        .setAttr(dnnl::graph::op::attr::pads_end, Operator::Ints, 4)
        .setAttr(dnnl::graph::op::attr::dilations, Operator::Ints, 5)
        .setAttr(dnnl::graph::op::attr::groups, Operator::Int, 6)
        .setAttr(dnnl::graph::op::attr::weights_format, std::string("OIX"))
        .setAttr(dnnl::graph::op::attr::data_format, std::string("NCX"));
  } else if (
      (nodeKind == Symbol::fromQualString("aten::_convolution")) ||
      (nodeKind == Symbol::fromQualString("aten::convolution"))) {
    bool transposed = toIValue(node->namedInput("transposed"))->toBool();
    REQUIRE(!transposed);
    return Operator(node, opkind::Convolution)
        .setInput(0, 1, 2)
        .setOutput(dnnl_graph_, 0)
        .setAttr(dnnl::graph::op::attr::strides, Operator::Ints, 3)
        .setAttr(dnnl::graph::op::attr::pads_begin, Operator::Ints, 4)
        .setAttr(dnnl::graph::op::attr::pads_end, Operator::Ints, 4)
        .setAttr(dnnl::graph::op::attr::dilations, Operator::Ints, 5)
        .setAttr(dnnl::graph::op::attr::groups, Operator::Int, 8)
        .setAttr(dnnl::graph::op::attr::weights_format, std::string("OIX"))
        .setAttr(dnnl::graph::op::attr::data_format, std::string("NCX"));
  } else if (nodeKind == Symbol::fromQualString("aten::batch_norm")) {
    auto training = toIValue(node->namedInput("training"));
    REQUIRE(training.has_value()); // cannot get training status in script mode
    if (!training->toBool()) {
      return Operator(node, opkind::BatchNormInference)
          .setInput(0, 1, 2, 3, 4)
          .setOutput(dnnl_graph_, 0)
          .setAttr(dnnl::graph::op::attr::epsilon, Operator::Float, 7)
          .setAttr(dnnl::graph::op::attr::data_format, std::string("NCX"));
    }
  } else if (nodeKind == Symbol::fromQualString("aten::layer_norm")) {
    auto normalized_shape = toIValue(node->namedInput("normalized_shape"));
    REQUIRE(normalized_shape->toIntList().size() == 1);
    return Operator(node, opkind::LayerNorm)
        .setInput(0, 2, 3)
        .setOutput(dnnl_graph_, 0)
        .setAttr(dnnl::graph::op::attr::epsilon, Operator::Float, 4)
        .setAttr(dnnl::graph::op::attr::keep_stats, false);
  } else if (nodeKind == Symbol::fromQualString("aten::addmm")) {
    auto alpha = toIValue(node->namedInput("alpha"));
    auto beta = toIValue(node->namedInput("beta"));
    if (alpha.has_value() && beta.has_value()) {
      if ((alpha->toDouble() == 1.0) && (beta->toDouble() == 1.0)) {
        return Operator(node, opkind::MatMul)
            .setInput(1, 2, 0)
            .setOutput(dnnl_graph_, 0);
      } else if ((alpha->toDouble() == 1.0) && (beta->toDouble() == 0.0)) {
        return Operator(node, opkind::MatMul)
            .setInput(1, 2)
            .setOutput(dnnl_graph_, 0);
      }
    }
  } else if (nodeKind == Symbol::fromQualString("aten::add"))
    return makeBinaryOp(node, opkind::Add);
  else if (nodeKind == Symbol::fromQualString("aten::mul"))
    return makeBinaryOp(node, opkind::Multiply);
  else if (nodeKind == Symbol::fromQualString("aten::div"))
    return makeBinaryOp(node, opkind::Divide);
  else if (nodeKind == Symbol::fromQualString("aten::tanh"))
    return makeEltwiseOp(node, opkind::Tanh);
  else if (nodeKind == Symbol::fromQualString("aten::relu"))
    return makeEltwiseOp(node, opkind::ReLU);
  else if (nodeKind == Symbol::fromQualString("aten::elu"))
    return makeEltwiseOp(node, opkind::Elu)
        .setAttr(dnnl::graph::op::attr::alpha, Operator::Float, 1);
  else if (nodeKind == Symbol::fromQualString("aten::sigmoid"))
    return makeEltwiseOp(node, opkind::Sigmoid);
  else if (nodeKind == Symbol::fromQualString("aten::gelu"))
    return makeEltwiseOp(node, opkind::GELU);
  else if (nodeKind == Symbol::fromQualString("aten::round"))
    return makeEltwiseOp(node, opkind::Round);
  else if (nodeKind == Symbol::fromQualString("aten::exp"))
    return makeEltwiseOp(node, opkind::Exp);
  else if (nodeKind == Symbol::fromQualString("aten::sqrt"))
    return makeEltwiseOp(node, opkind::Sqrt);
  else if (nodeKind == Symbol::fromQualString("aten::abs"))
    return makeEltwiseOp(node, opkind::Abs);
  else if (nodeKind == Symbol::fromQualString("aten::square"))
    return makeEltwiseOp(node, opkind::Square);
  else if (nodeKind == Symbol::fromQualString("aten::clamp")) {
    // PyTorch API already checks that both min & max are not None.
    // But we can check it nevertheless.
    auto clamp_min = toIValue(node->input(1));
    auto clamp_max = toIValue(node->input(2));
    REQUIRE(!(clamp_max->isNone() && clamp_min->isNone()));
    auto clamp_min_value = (clamp_min->isNone())
        ? -std::numeric_limits<float>::infinity()
        : Operator::ScalarToFloat(node, 1);
    auto clamp_max_value = (clamp_max->isNone())
        ? std::numeric_limits<float>::infinity()
        : Operator::ScalarToFloat(node, 2);
    return makeEltwiseOp(node, opkind::Clamp)
        .setAttr(dnnl::graph::op::attr::min, clamp_min_value)
        .setAttr(dnnl::graph::op::attr::max, clamp_max_value);
  } else if (nodeKind == Symbol::fromQualString("aten::hardtanh")) {
    return makeEltwiseOp(node, opkind::Clamp)
        .setAttr(dnnl::graph::op::attr::min, Operator::ScalarToFloat, 1)
        .setAttr(dnnl::graph::op::attr::max, Operator::ScalarToFloat, 2);
  } else if (nodeKind == Symbol::fromQualString("aten::hardswish"))
    return makeEltwiseOp(node, opkind::HardSwish);
  else if (nodeKind == Symbol::fromQualString("aten::log"))
    return makeEltwiseOp(node, opkind::Log);
  else if (nodeKind == Symbol::fromQualString("aten::leaky_relu")) {
    return makeEltwiseOp(node, opkind::LeakyReLU)
        .setAttr(dnnl::graph::op::attr::alpha, Operator::Float, 1);
  } else if (nodeKind == Symbol::fromQualString("aten::relu6")) {
    return makeEltwiseOp(node, opkind::Clamp)
        .setAttr(dnnl::graph::op::attr::min, 0.f)
        .setAttr(dnnl::graph::op::attr::max, 6.f);
  } else if (
      (nodeKind == Symbol::fromQualString("aten::softmax")) ||
      (nodeKind == Symbol::fromQualString("aten::_softmax"))) {
    auto axis = toIValue(node->namedInput("dim"))->toInt();
    return Operator(node, opkind::SoftMax)
        .setInput(0)
        .setOutput(dnnl_graph_, 0)
        .setAttr(dnnl::graph::op::attr::axis, axis);
  } else if (nodeKind == Symbol::fromQualString("aten::_log_softmax")) {
    auto axis = toIValue(node->namedInput("dim"))->toInt();
    return Operator(node, opkind::LogSoftmax)
        .setInput(0)
        .setOutput(dnnl_graph_, 0)
        .setAttr(dnnl::graph::op::attr::axis, axis);
  } else if (nodeKind == Symbol::fromQualString("aten::cat")) {
    auto o = Operator(node, opkind::Concat);
    REQUIRE(node->namedInput("tensors")->node()->kind() == prim::ListConstruct);
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
    return o.setOutput(dnnl_graph_, 0)
        .setAttr(dnnl::graph::op::attr::axis, Operator::Int, 1);
  } else if (
      (nodeKind == Symbol::fromQualString("aten::max_pool2d")) ||
      (nodeKind == Symbol::fromQualString("aten::max_pool2d_with_indices"))) {
    // Currently, LLGA lacks support to create indices mask.
    // Once it's supported, max_pool2d_with_indices should be mapped differently
    REQUIRE(node->namedInput("kernel_size")->node()->kind() == prim::Constant);
    auto rounding_type =
        toIValue(node->namedInput("ceil_mode"))->toBool() ? "ceil" : "floor";
    return Operator(node, opkind::MaxPool)
        .setInput(0)
        .setOutput(dnnl_graph_, 0)
        .setAttr(dnnl::graph::op::attr::kernel, Operator::Ints, 1)
        .setAttr(dnnl::graph::op::attr::strides, Operator::Ints, 2)
        .setAttr(dnnl::graph::op::attr::pads_begin, Operator::Ints, 3)
        .setAttr(dnnl::graph::op::attr::pads_end, Operator::Ints, 3)
        .setAttr(dnnl::graph::op::attr::dilations, Operator::Ints, 4)
        .setAttr(
            dnnl::graph::op::attr::rounding_type, std::string(rounding_type))
        .setAttr(dnnl::graph::op::attr::data_format, std::string("NCX"));
  } else if (nodeKind == Symbol::fromQualString("aten::avg_pool2d")) {
    // TODO: do we need add checks for all Constants?
    REQUIRE(node->namedInput("kernel_size")->node()->kind() == prim::Constant);
    auto rounding_type =
        toIValue(node->namedInput("ceil_mode"))->toBool() ? "ceil" : "floor";
    auto divisor_override = toIValue(node->namedInput("divisor_override"));
    REQUIRE(divisor_override->isNone());
    return Operator(node, opkind::AvgPool)
        .setInput(0)
        .setOutput(dnnl_graph_, 0)
        .setAttr(dnnl::graph::op::attr::kernel, Operator::Ints, 1)
        .setAttr(dnnl::graph::op::attr::strides, Operator::Ints, 2)
        .setAttr(dnnl::graph::op::attr::pads_begin, Operator::Ints, 3)
        .setAttr(dnnl::graph::op::attr::pads_end, Operator::Ints, 3)
        .setAttr(dnnl::graph::op::attr::exclude_pad, !Operator::Bool(node, 5))
        .setAttr(
            dnnl::graph::op::attr::rounding_type, std::string(rounding_type))
        .setAttr(dnnl::graph::op::attr::data_format, std::string("NCX"));
  } else if (nodeKind == Symbol::fromQualString("aten::matmul")) {
    auto dim0 = getDimensions(node->namedInput("self")).value_or(-1);
    auto dim1 = getDimensions(node->namedInput("other")).value_or(-1);
    // TODO: support all shape combinations
    REQUIRE(
        (dim0 == 2 && dim1 == 2) || (dim0 == 4 && dim1 == 4) ||
        (dim0 == 3 && dim1 == 2));
    return Operator(node, opkind::MatMul)
        .setInput(0, 1)
        .setOutput(dnnl_graph_, 0);
  } // fall through
  else if (nodeKind == Symbol::fromQualString("aten::mm")) {
    return Operator(node, opkind::MatMul)
        .setInput(0, 1)
        .setOutput(dnnl_graph_, 0);
  } else if (nodeKind == Symbol::fromQualString("aten::bmm")) {
    return Operator(node, opkind::MatMul)
        .setInput(0, 1)
        .setOutput(dnnl_graph_, 0);
  } else if (nodeKind == Symbol::fromQualString("aten::linear")) {
    return Operator(node, opkind::MatMul)
        .setInput(0, 1, 2)
        .setOutput(dnnl_graph_, 0)
        .setAttr(dnnl::graph::op::attr::transpose_b, true);
  } else if (nodeKind == Symbol::fromQualString("aten::permute")) {
    REQUIRE(aliasDb_->hasInputWriters(node) == false);
    return Operator(node, opkind::StaticTranspose)
        .setInput(0)
        .setOutput(dnnl_graph_, 0)
        .setAttr(
            dnnl::graph::op::attr::order,
            toIValue(node->namedInput("dims"))->toIntVector());
  } else if (nodeKind == Symbol::fromQualString("aten::contiguous")) {
    // Contiguous should only be mapped to oneDNN Graph if the destination
    // memory-layout is different than the source memory-format
    // Strides would be different, but shape would be same
    auto typeOfInput = node->input(0)->type()->expect<TensorType>();
    auto typeOfOutput = node->output(0)->type()->expect<TensorType>();
    auto inputStrides = typeOfInput->strides().concrete_sizes();
    auto outputStrides = typeOfOutput->strides().concrete_sizes();
    REQUIRE(inputStrides != outputStrides);
    return Operator(node, opkind::Reorder)
        .setInput(0)
        .setOutput(dnnl_graph_, 0);
  }
  GRAPH_DEBUG("Making ", nodeKind.toQualString(), " a wildcard");
  return makeWildcardOp(node);
}

static DeviceType inferDeviceFromValue(Value* v) {
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

static DeviceType inferDevice(const std::shared_ptr<Graph>& graph) {
  auto dt = inferDeviceFromValue(graph->inputs()[0]);
  TORCH_CHECK(
      std::all_of(
          graph->inputs().begin(),
          graph->inputs().end(),
          [dt](Value* v) { return inferDeviceFromValue(v) == dt; }),
      "All inputs must have the same deive type");
  return dt;
}

static dnnl::engine::kind getLlgaEngineKind(DeviceType type) {
  switch (type) {
    case DeviceType::CPU:
      return dnnl::engine::kind::cpu;
    default:
      TORCH_CHECK(false, "Not support device type ", type);
  }
}

static void mayAddListConstructIntoConcatPartition(
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
    auto listConstruct = n->namedInput("tensors")->node();
    auto partitionId = opToOwningPartition.get(n);
    opToOwningPartition.add(listConstruct, partitionId);
  }
}

// Verify that input tensors are compatible with oneDNN Graph.
// Scalars would be converted to 1-D tensors later anyway,
// but they shouldn't be complex-double
// If this check fails, convert op to wildcard
static bool checkInputCompatibility(Node* node) {
  auto allInputs = node->inputs();
  for (auto input : allInputs) {
    c10::IValue inputIValue = toIValue(input);
    if (inputIValue.isTensor()) {
      const at::Tensor& tensor = inputIValue.toTensor();
      if (tensor.device() != at::kCPU) {
        return false;
      }
      auto dtype = tensor.scalar_type();
      if ((dtype != at::ScalarType::BFloat16) &&
          (dtype != at::ScalarType::Float) && (dtype != at::ScalarType::Long)) {
        // We've allowed Long dtype here although oneDNN Graph does not support
        // Long dtype because oneDNN Graph will end up not handling the op that
        // has an input with Long dtype, so it'd be handled by PyTorch.
        return false;
      }
    } else if (inputIValue.isScalar()) {
      if (inputIValue.isComplexDouble()) {
        return false;
      }
    } else if (input->type()->isSubtypeOf(TensorType::get())) {
      auto input_typeptr = input->type()->cast<TensorType>();
      if (input_typeptr->scalarType().has_value()) {
        at::ScalarType dtype = input_typeptr->scalarType().value();
        if ((dtype != at::ScalarType::Float) &&
            (dtype != at::ScalarType::BFloat16)) {
          return false;
        }
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
  dnnl_graph_ = std::make_unique<dnnl::graph::graph>(engineKind);
  aliasDb_ = std::make_unique<torch::jit::AliasDb>(graph);
  GRAPH_DEBUG("Constructing LLGA graph");
  // TODO: select nodes in top-level block for now
  for (auto* node : graph->block()->nodes()) {
    auto kindOfNode = node->kind();
    GRAPH_DEBUG("Trying to add ", kindOfNode.toQualString());
    if (checkInputCompatibility(node)) {
      auto op = createOperator(node);
      dnnl_graph_->add_op(op.llgaOp());
      GRAPH_DEBUG("  Added node ", kindOfNode.toQualString());
    } else {
      GRAPH_DEBUG("Incompatible inputs for ", kindOfNode.toQualString());
      dnnl_graph_->add_op(makeWildcardOp(node).llgaOp());
    }

    for (Value* input : node->inputs()) {
      tensorIdToValue_.emplace(input->unique(), input);
    }
  }

  dnnl_graph_->finalize();

  GRAPH_DEBUG("Get Partitions");
  std::vector<dnnl::graph::partition> partitions =
      dnnl_graph_->get_partitions(policy);
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
static bool isBetterSuitedForLLGA(NodeKind kindOfOp) {
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
  const auto num_output = n->is(attr::output_layouts).size();
  TORCH_CHECK(
      offset < num_output,
      "Out of range. (Invalid index ",
      offset,
      " for attr::output_layouts with size ",
      num_output,
      ")");
  auto& layouts =
      const_cast<std::vector<int64_t>&>(n->is(attr::output_layouts)); // NOLINT
  layouts.at(offset) = OPAQUE_LAYOUT;
}

bool LlgaNodeWrapper::useOpaqueLayout(size_t offset) const {
  const auto num_output = n->is(attr::output_layouts).size();
  TORCH_CHECK(
      offset < num_output,
      "Out of range. (Invalid index ",
      offset,
      " for attr::output_layouts with size ",
      num_output,
      ")");
  return n->is(attr::output_layouts)[offset] == OPAQUE_LAYOUT;
}

} // namespace torch::jit::fuser::onednn
