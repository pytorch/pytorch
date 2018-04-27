#include "caffe2/opt/converter.h"
#include "caffe2/opt/fusion.h"

namespace caffe2 {
namespace opt {

using namespace nom;

template <typename OperationT, typename ActivationT, typename FusedT>
bool fusionHelper(std::string fusedName, repr::NNGraph* g) {
  for (auto& node_pair : repr::nn::dataIterator<OperationT>(*g)) {
    repr::NNGraph::NodeRef node;
    OperationT* operation;
    std::tie(operation, node) = node_pair;

    // Single output check (intrinsic to a operation, but we double check)
    auto outputs = repr::nn::getOutputs(node);
    if (outputs.size() != 1) {
      continue;
    }
    auto tensorNode = outputs.front();

    // Single user check.
    auto consumers = repr::nn::getConsumers(tensorNode);
    if (consumers.size() != 1) {
      continue;
    }

    // Followed by Activation check.
    auto* nextNode = consumers.front();
    if (!repr::nn::is<ActivationT>(nextNode)) {
      continue;
    }

    // Naming for operationenience
    auto* operationNode = node;
    auto* reluNode = nextNode;

    // Create our Operation + Activation and annotate it by modifying the
    // original Operation
    auto* fusedNode = g->createNode(util::make_unique<FusedT>(*operation));
    auto fused = repr::nn::get<FusedT>(fusedNode);

    // Modification of the original Fuseable
    auto oldAnnotation = operation->getMutableAnnotation();
    if (oldAnnotation) {
      if (isa<caffe2::Caffe2Annotation>(oldAnnotation)) {
        fused->setAnnotation(util::make_unique<caffe2::Caffe2Annotation>());
        auto annotation = dyn_cast<caffe2::Caffe2Annotation>(fused->getMutableAnnotation());
        auto operationOp = dyn_cast<caffe2::Caffe2Annotation>(oldAnnotation)->getMutableOperatorDef();
        operationOp->set_type(fusedName);
        annotation->setOperatorDef(operationOp);
      } else {
        assert(0 && "Unsupported annotation.");
      }
    }

    for (const auto input : repr::nn::getInputs(operationNode)) {
      g->createEdge(input, fusedNode);
    }
    for (const auto output : repr::nn::getOutputs(operationNode)) {
      g->createEdge(fusedNode, output);
    }

    g->deleteNode(operationNode);
    g->deleteNode(tensorNode);
    g->deleteNode(reluNode);

    return true;
  }
  return false;
}

bool fuseConvRelu(nom::repr::NNModule* nn) {
  while (fusionHelper<repr::Conv, repr::Relu, repr::ConvRelu>(
      "ConvRelu", &nn->dataFlow)) {
  }
  return true;
}

bool fuseAveragePoolRelu(nom::repr::NNModule* nn) {
  while (fusionHelper<repr::AveragePool, repr::Relu, repr::AveragePoolRelu>(
      "AveragePoolRelu", &nn->dataFlow)) {
  }
  return true;
}

bool fuseMaxPoolRelu(nom::repr::NNModule* nn) {
  while (fusionHelper<repr::MaxPool, repr::Relu, repr::MaxPoolRelu>(
      "MaxPoolRelu", &nn->dataFlow)) {
  }
  return true;
}

bool fuseSumRelu(nom::repr::NNModule* nn) {
  while (fusionHelper<repr::Sum, repr::Relu, repr::SumRelu>(
      "SumRelu", &nn->dataFlow)) {
  }
  return true;
}

} // namespace opt
} // namespace caffe2
