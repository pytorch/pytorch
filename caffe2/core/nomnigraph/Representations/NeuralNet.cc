#include "nomnigraph/Representations/NeuralNet.h"
#include "nomnigraph/Graph/Algorithms.h"

namespace nom {
namespace repr {

// NOLINTNEXTLINE(modernize-use-equals-default)
NeuralNetOperator::~NeuralNetOperator() {}

const std::string NeuralNetOperator::getName() const {
  switch (getKind()) {
#include "nomnigraph/Generated/OpNames.h"
    case NNKind::While:
      return "While";
    case NNKind::NNPhi:
      return "Phi";
    case NNKind::GenericOperator:
      return dyn_cast<GenericOperator>(this)->getName();
    default:
      return "Unknown";
  }
}

// NOLINTNEXTLINE(modernize-use-equals-default)
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

NNGraph::NodeRef NNModule::createUniqueDataNode(const std::string& s) {
  auto curr_name = s;
  auto iter = 0;
  bool need_name = true;
  do {
    need_name = false;
    for (const auto& node : dataFlow.getMutableNodes()) {
      if (nn::getName(node) == curr_name) {
        std::stringstream ss;
        ss << iter;
        curr_name = s + "_" + ss.str();
        iter++;
        need_name = true;
        break;
      }
    }
  } while (need_name);
  return dataFlow.createNode(std::make_unique<nom::repr::Tensor>(curr_name));
}

void NNModule::replaceSubgraph(
    const NNSubgraph& subgraph,
    const NNGraph::NodeRef& node,
    const std::vector<NNGraph::NodeRef>& node_inputs,
    const std::vector<NNGraph::NodeRef>& node_outputs) {
  auto sg = subgraph;
  auto sg_inputs = nn::getInputs(sg);
  auto sg_outputs = nn::getOutputs(sg);

  auto sg_inputs_copy = sg_inputs;
  auto sg_outputs_copy = sg_outputs;

  for (const auto& input : node_inputs) {
    sg_inputs_copy.erase(input);
    // outputs may contain inputs that have additional
    // consumers external to the subgraph
    sg_outputs_copy.erase(input);
  }
  assert(sg_inputs_copy.size() == 0 && "Not all inputs were listed");

  for (const auto& output : node_outputs) {
    sg_outputs_copy.erase(output);
  }
  assert(sg_outputs_copy.size() == 0 && "Not all outputs were listed");

  for (auto& input : node_inputs) {
    dataFlow.createEdge(input, node);
    sg.removeNode(input);
  }
  for (auto& output : node_outputs) {
    if (sg_inputs.count(output)) {
      dataFlow.createEdge(node, createUniqueDataNode());
      continue;
    }
    dataFlow.createEdge(node, output);
    sg.removeNode(output);
  }
  deleteSubgraph(sg);
}

void NNModule::deleteSubgraph(const NNSubgraph& subgraph) {
  dataFlow.deleteNodes(subgraph.getNodes());
}

namespace nn {

bool hasProducer(NNGraph::NodeRef n) {
  return n->getInEdges().size() != 0;
}

NNGraph::NodeRef getProducer(NNGraph::NodeRef n) {
  assert(
      is<NeuralNetData>(n) &&
      "getProducer only works with NeuralNetData types.");
  auto inEdges = n->getInEdges();
  assert(inEdges.size() > 0 && "Tensor does not have a producer.");
  assert(
      inEdges.size() == 1 &&
      "Malformed NNGraph, NeuralNetData has multiple producers.");
  return inEdges.front()->tail();
}

bool hasConsumer(NNGraph::NodeRef n) {
  return n->getOutEdges().size() != 0;
}

std::vector<NNGraph::NodeRef> getConsumers(NNGraph::NodeRef n) {
  assert(
      is<NeuralNetData>(n) &&
      "getProducer only works with NeuralNetData types.");
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
  assert(
      is<NeuralNetOperator>(n) &&
      "getInputs only works with NeuralNetOperator types.");
  std::vector<NNGraph::NodeRef> out;
  for (auto inEdge : n->getInEdges()) {
    out.emplace_back(inEdge->tail());
  }
  return out;
}

std::vector<NNGraph::NodeRef> getOutputs(NNGraph::NodeRef n) {
  assert(
      is<NeuralNetOperator>(n) &&
      "getOutputs only works with NeuralNetOperator types.");
  std::vector<NNGraph::NodeRef> out;
  for (auto outEdge : n->getOutEdges()) {
    out.emplace_back(outEdge->head());
  }
  return out;
}

std::string getName(NNGraph::NodeRef n) {
  if (is<NeuralNetData>(n)) {
    return nn::get<NeuralNetData>(n)->getName();
  } else if (is<NeuralNetOperator>(n)) {
    return nn::get<NeuralNetOperator>(n)->getName();
  }
  return "Unknown";
}

std::set<NNGraph::NodeRef> getInputs(const NNSubgraph& subgraph) {
  std::set<NNGraph::NodeRef> subgraph_inputs;
  for (const auto& node : subgraph.getNodes()) {
    NOM_REQUIRE_OR_CONT(is<NeuralNetData>(node));
    if (hasProducer(node)) {
      if (!subgraph.hasNode(getProducer(node))) {
        subgraph_inputs.insert(node);
      }
    } else {
      subgraph_inputs.insert(node);
    }
  }
  return subgraph_inputs;
}

std::set<NNGraph::NodeRef> getOutputs(const NNSubgraph& subgraph) {
  std::set<NNGraph::NodeRef> subgraph_outputs;
  for (const auto& n : subgraph.getNodes()) {
    NOM_REQUIRE_OR_CONT(is<NeuralNetData>(n));
    if (hasConsumer(n)) {
      for (const auto& consumer : getConsumers(n)) {
        if (!subgraph.hasNode(consumer)) {
          subgraph_outputs.insert(n);
        }
      }
    } else {
      subgraph_outputs.insert(n);
    }
  }
  return subgraph_outputs;
}

void replaceProducer(
    NNGraph::NodeRef tensorNode,
    NNGraph::NodeRef newProducer) {
  assert(
      is<NeuralNetData>(tensorNode) &&
      "First argument must contain NeuralNetData");
  auto inEdges = tensorNode->getInEdges();
  assert(
      inEdges.size() == 1 && "Tensor node passed in does not have a producer");
  auto edge = inEdges.at(0);
  auto prevProducer = edge->tail();
  prevProducer->removeOutEdge(edge);
  edge->setTail(newProducer);
  newProducer->addOutEdge(edge);
}

void replaceAllUsesWith(
    NNGraph::NodeRef oldTensorNode,
    NNGraph::NodeRef newTensorNode) {
  const auto edges = oldTensorNode->getOutEdges();
  for (const auto& edge : edges) {
    edge->setTail(newTensorNode);
    oldTensorNode->removeOutEdge(edge);
    newTensorNode->addOutEdge(edge);
  }
}

void replaceAsConsumer(
    NNGraph::NodeRef oldConsumer,
    NNGraph::NodeRef newConsumer) {
  const auto edges = oldConsumer->getInEdges();
  for (const auto& edge : edges) {
    edge->setHead(newConsumer);
    oldConsumer->removeInEdge(edge);
    newConsumer->addInEdge(edge);
  }
}

NNGraph::NodeRef
createOutput(NNModule* nn, NNGraph::NodeRef producer, std::string name) {
  auto outputNode =
      nn->dataFlow.createNode(std::make_unique<nom::repr::Tensor>(name));
  nn->dataFlow.createEdge(producer, outputNode);
  return outputNode;
}

// Get all nodes tracked by CF graph
static std::unordered_set<repr::NNGraph::NodeRef> getTrackedNodes(
    repr::NNCFGraph& cf) {
  std::unordered_set<repr::NNGraph::NodeRef> cfTrackedNodes;
  for (const auto& bbNode : cf.getMutableNodes()) {
    auto& bb = bbNode->data();
    for (const auto node : bb.getInstructions()) {
      cfTrackedNodes.insert(node);
    }
  }
  return cfTrackedNodes;
}

static size_t coalesceInsertedDataDependenciesHelper(repr::NNModule* m) {
  auto cfTrackedNodes = getTrackedNodes(m->controlFlow);

  for (auto& bbNode : m->controlFlow.getMutableNodes()) {
    auto bb = bbNode->mutableData();
    // We mutate the instructions of the bb, so we copy here.
    // TODO make this an iterator and simply promote it on insertion.
    auto instrsCopy = bb->getInstructions();
    for (const auto instr : instrsCopy) {
      for (const auto input : repr::nn::getInputs(instr)) {
        if (!repr::nn::hasProducer(input)) {
          continue;
        }
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

  // Now we track new nodes that have no relationship to the old CFGraph
  auto cfTrackedNodes = getTrackedNodes(m->controlFlow);
  std::unordered_set<repr::NNGraph::NodeRef> dfNodes;
  for (auto node : m->dataFlow.getMutableNodes()) {
    if (repr::nn::is<NeuralNetOperator>(node) && !cfTrackedNodes.count(node)) {
      dfNodes.insert(node);
    }
  }

  auto newBbNode = m->controlFlow.createAnonymousFunction();
  auto sccs = algorithm::tarjans(&m->dataFlow);
  for (auto iter = sccs.rbegin(); iter != sccs.rend(); ++iter) {
    for (auto node : iter->getNodes()) {
      if (dfNodes.count(node)) {
        auto currentBasicBlock = newBbNode->mutableData();
        currentBasicBlock->pushInstructionNode(node);
      }
    }
  }

  // Finally we reconcile any data dependency issues (if we can).
  for (auto& bbNode : m->controlFlow.getMutableNodes()) {
    auto bb = bbNode->mutableData();
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int permutation;
    do {
      permutation = 0;
      std::unordered_set<repr::NNGraph::NodeRef> seen;
      for (auto instr_iter = bb->getMutableInstructions()->begin();
           instr_iter != bb->getMutableInstructions()->end();
           ++instr_iter) {
        // This cannot be auto& because *iter is pure R-ref
        auto instr = *instr_iter;
        for (auto& output : getOutputs(instr)) {
          for (auto& consumer : getConsumers(output)) {
            if (seen.count(consumer)) {
              bb->moveInstructionBefore(instr, consumer);
              ++permutation;
            }
          }
        }
        seen.insert(instr);
      }
    } while (permutation);
  }
}

bool hasSingleOutputAndConsumer(NNGraph::NodeRef nodeRef) {
  auto nodeOutputs = nn::getOutputs(nodeRef);
  NOM_REQUIRE_OR_RET_FALSE(nodeOutputs.size() == 1);
  auto nodeConsumers = nn::getConsumers(nodeOutputs.front());
  return nodeConsumers.size() == 1;
}

bool hasUniqueConsumer(NNGraph::NodeRef nodeRef) {
  auto nodeOutputs = nn::getOutputs(nodeRef);
  NNGraph::NodeRef nodeConsumer = nullptr;
  for (auto nodeOutput : nodeOutputs) {
    for (auto consumer : nn::getConsumers(nodeOutput)) {
      if (nodeConsumer && consumer && consumer != nodeConsumer) {
        return false;
      }
      nodeConsumer = consumer;
    }
  }
  return true;
}

NNMatchPredicate matchExternalTensorNode() {
  return NNMatchPredicate(nn::is<Tensor>).nonTerminal().excludeFromSubgraph();
}

} // namespace nn

} // namespace repr
} // namespace nom
