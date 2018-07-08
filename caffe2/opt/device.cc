#include "caffe2/opt/device.h"
#include "caffe2/core/logging.h"
#include "nomnigraph/Graph/Algorithms.h"

using namespace nom;
using namespace nom::repr;

std::vector<NNGraph::EdgeRef> getInputEdges(
    const NNGraph::SubgraphType& sg,
    const NNGraph& g) {
  std::vector<NNGraph::EdgeRef> inputTensorEdges;
  for (const auto& node : sg.Nodes) {
    NOM_REQUIRE_OR_CONT(nn::is<NeuralNetOperator>(node));
    NOM_REQUIRE_OR_CONT(nn::hasInputs(node));

    // Check if tensor's parents are in the sg
    for (const auto& input : nn::getInputs(node)) {
      NOM_REQUIRE_OR_CONT(
          !nn::hasProducer(input) ||
          sg.Nodes.count(nn::getProducer(input)) == 0);
      inputTensorEdges.emplace_back(g.getEdge(input, node));
    }
  }
  return inputTensorEdges;
}

std::vector<NNGraph::EdgeRef> getOutputEdges(
    const NNGraph::SubgraphType& sg,
    const NNGraph& g) {
  std::vector<NNGraph::EdgeRef> outputTensorEdges;
  for (const auto& node : sg.Nodes) {
    NOM_REQUIRE_OR_CONT(nn::is<NeuralNetOperator>(node));

    for (const auto& output : nn::getOutputs(node)) {
      auto consumers = nn::getConsumers(output);
      for (const auto& consumer : consumers) {
        NOM_REQUIRE_OR_CONT(sg.Nodes.count(consumer) == 0);
        outputTensorEdges.emplace_back(g.getEdge(node, output));
      }
      NOM_REQUIRE_OR_CONT(consumers.size() == 0);
      outputTensorEdges.emplace_back(g.getEdge(node, output));
    }
  }
  return outputTensorEdges;
}

namespace caffe2 {
namespace opt {

void insertCopies(
    NNModule* nn,
    std::function<bool(NNGraph::NodeRef)> supported,
    std::function<NNGraph::NodeRef(NNGraph&)> copyToFn,
    std::function<NNGraph::NodeRef(NNGraph&)> copyFromFn) {
  auto matches = nom::algorithm::binaryMatch(&nn->dataFlow, supported);

  // We're doing a lot of inplace mutation so this is necessary.
  std::set<NNGraph::EdgeRef> changedEdges;

  for (const auto& match : matches) {
    for (const auto& edge : getInputEdges(match, nn->dataFlow)) {
      NOM_REQUIRE_OR_CONT(changedEdges.count(edge) == 0);
      auto input = edge->tail();

      // This may be redundant, but we need the user's function
      // to get the type of the node they're using.
      auto copyNode = copyToFn(nn->dataFlow);
      bool redundant = false;
      // Rectify redudancies.
      for (const auto& consumer : nn::getConsumers(input)) {
        auto copyOp = nn::get<NeuralNetOperator>(copyNode);
        auto consumerOp = nn::get<NeuralNetOperator>(consumer);
        // We already have a copy node, let's reuse it.
        if (consumerOp->getKind() == copyOp->getKind()) {
          nn->dataFlow.deleteNode(copyNode);
          copyNode = consumer;
          redundant = true;
        }
      }

      auto data = nn::get<NeuralNetData>(input);
      auto newInput = redundant
          ? nn::getOutputs(copyNode).front()
          : nn->dataFlow.createNode(
                util::make_unique<repr::Tensor>(data->getName() + "_"));
      if (!redundant) {
        nn->dataFlow.createEdge(input, copyNode);
        nn->dataFlow.createEdge(copyNode, newInput);
      }

      input->removeOutEdge(edge);
      edge->setTail(newInput);

      changedEdges.insert(edge);
    }

    for (const auto& edge : getOutputEdges(match, nn->dataFlow)) {
      NOM_REQUIRE_OR_CONT(changedEdges.count(edge) == 0);
      auto output = edge->head();

      auto copyNode = copyFromFn(nn->dataFlow);
      auto data = nn::get<NeuralNetData>(output);
      auto newOutput = nn->dataFlow.createNode(
          util::make_unique<repr::Tensor>(data->getName() + "_"));

      output->removeInEdge(edge);
      edge->setHead(newOutput);

      changedEdges.insert(edge);

      nn->dataFlow.createEdge(newOutput, copyNode);
      nn->dataFlow.createEdge(copyNode, output);
    }
  }
}

} // namespace opt
} // namespace caffe2
