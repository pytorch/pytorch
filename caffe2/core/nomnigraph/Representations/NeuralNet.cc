#include "nomnigraph/Representations/NeuralNet.h"
#include "nomnigraph/Graph/Algorithms.h"

namespace nom {
namespace repr {

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

// Get all nodes tracked by CF graph
static std::unordered_set<repr::NNGraph::NodeRef> getTrackedNodes(
    repr::NNCFGraph& cf) {
  std::unordered_set<repr::NNGraph::NodeRef> cfTrackedNodes;
  for (const auto& bbNode : cf.getMutableNodes()) {
    auto bb = repr::nn::get<repr::BasicBlockType<repr::NNGraph>>(bbNode);
    for (const auto node : bb->getInstructions()) {
      cfTrackedNodes.insert(node);
    }
  }
  return cfTrackedNodes;
}

static size_t coalesceInsertedDataDependenciesHelper(repr::NNCFGraph& cfg) {
  auto cfTrackedNodes = getTrackedNodes(cfg);

  for (auto& bbNode : cfg.getMutableNodes()) {
    auto bb = repr::nn::get<repr::BasicBlockType<repr::NNGraph>>(bbNode);
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

void coalesceInsertedDataDependenciesGraphImpl(
    repr::NNGraph& dfg,
    repr::NNCFGraph& cfg) {
  size_t oldSize = 0;
  size_t newSize = 0;
  do {
    oldSize = newSize;
    newSize = coalesceInsertedDataDependenciesHelper(cfg);
  } while (newSize != oldSize);

  // Now we track new nodes that have no relationship to the old CFGraph
  auto cfTrackedNodes = getTrackedNodes(cfg);
  std::unordered_set<repr::NNGraph::NodeRef> dfNodes;
  for (auto node : dfg.getMutableNodes()) {
    if (repr::nn::is<NeuralNetOperator>(node) && !cfTrackedNodes.count(node)) {
      dfNodes.insert(node);
    }
  }

  auto newBbNode =
      cfg.createNode(util::make_unique<repr::BasicBlockType<repr::NNGraph>>());
  auto sccs = algorithm::tarjans(&dfg);
  for (auto iter = sccs.rbegin(); iter != sccs.rend(); ++iter) {
    for (auto node : iter->getNodes()) {
      if (dfNodes.count(node)) {
        auto currentBasicBlock = newBbNode->mutableData()->get();
        currentBasicBlock->pushInstructionNode(node);
      }
    }
  }

  // Finally we reconcile any data dependency issues (if we can).
  for (auto& bbNode : cfg.getMutableNodes()) {
    auto bb = bbNode->mutableData()->get();
    std::unordered_set<repr::NNGraph::NodeRef> seen;
    for (auto instr_iter = bb->getInstructions().begin();
         instr_iter != bb->getInstructions().end();
         ++instr_iter) {
      // This cannot be auto&, TODO figure out why
      auto instr = *instr_iter;
      for (auto& output : getOutputs(instr)) {
        for (auto& consumer : getConsumers(output)) {
          if (seen.count(consumer)) {
            bb->moveInstructionBefore(instr, consumer);
          }
        }
      }
      seen.insert(instr);
    }
  }
}

// TODO: move this to more generic location.
// TODO: [algo] improve this algorithm, as it is horrendously inefficient.
void coalesceInsertedDataDependencies(repr::NNModule* m) {
  return coalesceInsertedDataDependenciesGraphImpl(m->dataFlow, m->controlFlow);
}

NodeIteratorVector iterate(NNGraph& g) {
  auto& data_nodes = g.getMutableNodes();

  vector<std::reference_wrapper<NNGraph::NodeObj>> nodes;
  for (auto& node : data_nodes) {
    nodes.emplace_back(std::ref(*node));
  }
  auto is_in_graph = [&g](NNGraph::NodeRef node) {
    auto all_nodes = g.getMutableNodes();
    return std::find(
               all_nodes.begin(), all_nodes.end(), NNGraph::NodeRef(node)) !=
        all_nodes.end();
  };
  return NodeIteratorVector(nodes, is_in_graph);
}

NodeIteratorVector iterate(NNCFGraph& cfg, NNGraph& g) {
  coalesceInsertedDataDependenciesGraphImpl(g, cfg);
  vector<std::reference_wrapper<NNGraph::NodeObj>> nodes;
  auto topo_cfg = algorithm::tarjans(&cfg);
  for (auto iter = topo_cfg.rbegin(); iter != topo_cfg.rend(); ++iter) {
    assert(
        (iter->getNodes().size() == 1) &&
        "Control flow consists of loop, which is not supported now.");
    for (auto& bbNode : iter->getNodes()) {
      auto instr_copy = bbNode->data().get()->getInstructions();
      for (auto instr : instr_copy) {
        nodes.emplace_back(std::ref(*instr));
      }
    }
  }
  auto is_in_graph = [&cfg](NNGraph::NodeRef node) {
    for (auto& bbNode : cfg.getMutableNodes()) {
      if (bbNode->data().get()->hasInstruction(NNGraph::NodeRef(node))) {
        return true;
      }
    }
    return false;
  };
  return NodeIteratorVector(nodes, is_in_graph);
}

} // namespace nn

} // namespace repr
} // namespace nom
