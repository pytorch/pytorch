#include "nomnigraph/Transformations/ConnectNet.h"

namespace nom {
namespace transformations {

using namespace repr;

const std::string getDeviceFromNode(const NNGraph::NodeRef &node) {
  auto nnOp = nn::get<NeuralNetOperator>(node);
  const Annotation *annotation = nnOp->getAnnotation();
  if (annotation && isa<DeviceAnnotation>(annotation)) {
    auto device_annotation = dyn_cast<DeviceAnnotation>(annotation);
    return device_annotation->getDevice();
  }
  return "";
}

bool connectNet(NNGraph *g) {
  // Iterate through all the tensors in the graph.
  for (auto tensor_node_pair : nn::dataIterator<NeuralNetData>(*g)) {

    NNGraph::NodeRef tensorNode;
    NeuralNetData* tensor;
    std::tie(tensor, tensorNode) = tensor_node_pair;

    // This is an edge case for when a tensor is created from outside
    // the execution graph.
    if (!nn::hasProducer(tensorNode)) { continue; }

    auto producerDevice = getDeviceFromNode(nn::getProducer(tensorNode));
    for (auto& consumerNode : nn::getConsumers(tensorNode)) {

        auto consumerDevice = getDeviceFromNode(consumerNode);
        if (consumerDevice == producerDevice) { continue; }

        auto sendNode = g->createNode(util::make_unique<Send>());
        g->createEdge(tensorNode, sendNode);

        auto sendTensorNode = g->createNode(
            util::make_unique<Tensor>(tensor->getName() + "_send"));
        g->createEdge(sendNode, sendTensorNode);

        auto recvNode = g->createNode(util::make_unique<Receive>());
        g->createEdge(sendTensorNode, recvNode);

        auto recvTensorNode = g->createNode(
            util::make_unique<Tensor>(tensor->getName() + "_recv"));
        g->createEdge(recvNode, recvTensorNode);

        g->createEdge(recvTensorNode, consumerNode);

        // This is safe because we copied the edge list.
        g->deleteEdge(g->getEdge(tensorNode, consumerNode));
    }
  }
  return true;
}

} // namespace transformations
} // namespace nom
