#include "nomnigraph/Representations/NeuralNet.h"

namespace nom {
namespace repr {

NeuralNetOperator::~NeuralNetOperator() {}

const std::string NeuralNetOperator::getName() const {
  switch (getKind()) {
  case NNKind::Conv:
    return "Conv";
  case NNKind::Relu:
    return "Relu";
  case NNKind::Send:
    return "Send";
  case NNKind::Receive:
    return "Receive";
  case NNKind::While:
    return "While";
  case NNKind::NNPhi:
    return "Phi";
  case NNKind::ConvRelu:
    return "ConvRelu";
  case NNKind::DynamicInput:
    return "DynamicInput";
  case NNKind::GenericOperator:
    return dyn_cast<GenericOperator>(this)->getName();
  default:
    return "Unknown";
  }
}

NeuralNetData::~NeuralNetData() {}

const std::string NeuralNetData::getName() const {
  switch (getKind()) {
  case NNDataKind::Tensor: {
    return dyn_cast<Tensor>(this)->getName();
  }
  default:
    return "";
  }
}

namespace nn {

bool hasProducer(NNGraph::NodeRef n) {
  return n->getInEdges().size() != 0;
}

NNGraph::NodeRef getProducer(NNGraph::NodeRef n) {
  assert(is<NeuralNetData>(n) && "getProducer only works with NeuralNetData types.");
  auto inEdges = n->getInEdges();
  assert(inEdges.size() > 0 && "Tensor does not have a producer.");
  assert(inEdges.size() == 1 && "Malformed NNGraph, NeuralNetData has multiple producers.");
  return inEdges.front()->tail();
}

std::vector<NNGraph::NodeRef> getConsumers(NNGraph::NodeRef n) {
  assert(is<NeuralNetData>(n) && "getProducer only works with NeuralNetData types.");
  std::vector<NNGraph::NodeRef> out;
  for (auto outEdge : n->getOutEdges()) {
    out.emplace_back(outEdge->head());
  }
  return out;
}

bool hasInputs(NNGraph::NodeRef n) {
  return n->getInEdges().size() != 0;
}

std::vector<NNGraph::NodeRef> getInputs(NNGraph::NodeRef n) {
  assert(is<NeuralNetOperator>(n) && "getInputs only works with NeuralNetOperator types.");
  std::vector<NNGraph::NodeRef> out;
  for (auto inEdge : n->getInEdges()) {
    out.emplace_back(inEdge->tail());
  }
  return out;
}

std::vector<NNGraph::NodeRef> getOutputs(NNGraph::NodeRef n) {
  assert(is<NeuralNetOperator>(n) && "getOutputs only works with NeuralNetOperator types.");
  std::vector<NNGraph::NodeRef> out;
  for (auto outEdge : n->getOutEdges()) {
    out.emplace_back(outEdge->head());
  }
  return out;
}

size_t coalesceInsertedDataDependenciesHelper(repr::NNModule* m) {
  // Get all nodes tracked by CF graph
  std::unordered_set<repr::NNGraph::NodeRef> cfTrackedNodes;
  for (const auto &bbNode : m->controlFlow.getMutableNodes()) {
    auto bb = repr::nn::get<repr::BasicBlockType<repr::NNGraph>>(bbNode);
    for (const auto node : bb->getInstructions()) {
      cfTrackedNodes.insert(node);
    }
  }

  for (auto &bbNode : m->controlFlow.getMutableNodes()) {
    auto bb = repr::nn::get<repr::BasicBlockType<repr::NNGraph>>(bbNode);
    // We mutate the instructions of the bb, so we copy here.
    // TODO make this an iterator and simply promote it on insertion.
    auto instrsCopy = bb->getInstructions();
    for (const auto instr : instrsCopy) {
      for (const auto input : repr::nn::getInputs(instr)) {
        if (!repr::nn::hasProducer(input)) { continue; }
        auto producer = repr::nn::getProducer(input);
        if (!cfTrackedNodes.count(producer)) {
          bb->insertInstructionBefore(producer, instr);
          cfTrackedNodes.insert(producer);
        }
      }
    }
  }

  return cfTrackedNodes.size();
}

// TODO: move this to more generic location.
// TODO: [algo] improve this algorithm, as it is horrendously inefficient.
void coalesceInsertedDataDependencies(repr::NNModule* m) {
  size_t oldSize = 0;
  size_t newSize = 0;
  do {
    oldSize = newSize;
    newSize = coalesceInsertedDataDependenciesHelper(m);
  } while (newSize != oldSize);
}

} // namespace nn

} // namespace repr
} // namespace nom
