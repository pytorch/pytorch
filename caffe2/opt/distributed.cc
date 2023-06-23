#include "caffe2/opt/distributed.h"
#include "caffe2/opt/converter.h"

namespace caffe2 {

using namespace nom::repr;

static void setDeviceOption(NNGraph::NodeRef n, caffe2::DeviceOption& d) {
  getOrAddCaffe2Annotation(n);
  auto op = nn::get<NeuralNetOperator>(n);
  auto c2Annot = dyn_cast<caffe2::Caffe2Annotation>(op->getMutableAnnotation());
  CAFFE_ENFORCE(c2Annot, "getOrAddCaffe2Annotation failed!");
  c2Annot->setDeviceOption(d);
}

void addBlobDeviceOptions(
    std::map<std::string, caffe2::DeviceOption> blobMap,
    NNModule* nn) {
  // Names we've seen in the NNModule. Uniqueness within inputs or outputs is ensured
  // but same blob can exist across inputs and outputs
  std::unordered_set<std::string> seen_inputs;
  std::unordered_set<std::string> seen_outputs;
  std::unordered_set<std::string> seen;

  auto declareNodes = nn::filter<Declare>(*nn);

  for (auto& declareNode : declareNodes) {
    auto inputNode = nn::getOutputs(declareNode).at(0);
    auto input = nn::get<nom::repr::Tensor>(inputNode);

    if (!blobMap.count(input->getName())) {
      continue;
    }

    CAFFE_ENFORCE(
        !seen_inputs.count(input->getName()),
        "Ambiguous name->deviceOption map.  Please do this manually. Affected blob: " + input->getName());
    seen_inputs.insert(input->getName());
    seen.insert(input->getName());
    setDeviceOption(declareNode, blobMap[input->getName()]);
  }

  auto exportNodes = nn::filter<Export>(*nn);

  for (auto& exportNode : exportNodes) {
    auto outputNode = nn::getInputs(exportNode).at(0);
    auto output = nn::get<nom::repr::Tensor>(outputNode);

    if (!blobMap.count(output->getName())) {
      continue;
    }

    CAFFE_ENFORCE(
        !seen_outputs.count(output->getName()),
        "Ambiguous name->deviceOption map.  Please do this manually. Affected blob: " + output->getName());

    seen_outputs.insert(output->getName());
    seen.insert(output->getName());
    setDeviceOption(exportNode, blobMap[output->getName()]);
  }

  if (seen.size() != blobMap.size()) {
    std::ostringstream os;
    for (const auto& kv : blobMap) {
      if (!(seen.count(kv.first))) {
        os << "\"" << kv.first << "\" ";
      }
    }
    CAFFE_ENFORCE(
        seen.size() == blobMap.size(),
        "Unused names in the blob map: ",
        os.str());
  }
}

void injectDataEdgeIndicators(nom::repr::NNModule* nn) {
  for (auto& input : nn->inputs) {
    auto declareNode =
        nn->dataFlow.createNode(std::make_unique<Declare>());
    nn->dataFlow.createEdge(declareNode, input);
  }

  for (auto& output : nn->outputs) {
    auto exportNode = nn->dataFlow.createNode(std::make_unique<Export>());
    nn->dataFlow.createEdge(output, exportNode);
  }

  nn->inputs.clear();
  nn->outputs.clear();
}

void removeDataEdgeIndicators(nom::repr::NNModule* nn) {
  auto declareNodes = nn::filter<Declare>(*nn);
  for (auto& declareNode : declareNodes) {
    auto input = nn::getOutputs(declareNode).at(0);
    nn->inputs.insert(input);
    nn->dataFlow.deleteNode(declareNode);
  }
  auto exportNodes = nn::filter<Export>(*nn);
  for (auto& exportNode : exportNodes) {
    auto output = nn::getInputs(exportNode).at(0);
    nn->outputs.insert(output);
    nn->dataFlow.deleteNode(exportNode);
  }
}

nom::repr::NNModule convertToNNModule(
    caffe2::NetDef& net,
    std::map<std::string, caffe2::DeviceOption> blobMap) {
  auto nn = convertToNNModule(net);
  injectDataEdgeIndicators(&nn);
  addBlobDeviceOptions(blobMap, &nn);
  return nn;
}

} // namespace caffe2
