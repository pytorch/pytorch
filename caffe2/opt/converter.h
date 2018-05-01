#ifndef NOM_CONVERTERS_CAFFE2_H
#define NOM_CONVERTERS_CAFFE2_H

#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Representations/ControlFlow.h"
#include "nomnigraph/Representations/NeuralNet.h"
#include "caffe2/core/common.h"
#include "caffe2/proto/caffe2.pb.h"

#include <unordered_map>

namespace caffe2 {

class Caffe2Annotation : public nom::repr::Annotation {
public:
  Caffe2Annotation() : Annotation(AnnotationKind::Caffe2) {}
  Caffe2Annotation(std::string device)
      : Annotation(AnnotationKind::Caffe2), Device(device) {}

  void setDevice(std::string device) { Device = device; }
  const std::string getDevice() const { return Device; }

  void setOperatorDef(caffe2::OperatorDef* opDef) {
    OpDef = opDef;
  }
  const caffe2::OperatorDef* getOperatorDef() const { 
    assert(OpDef && "OperatorDef was never set.  Use Caffe2Annotation::setOperatorDef.");
    return OpDef;
  }
  caffe2::OperatorDef* getMutableOperatorDef() { 
    assert(OpDef && "OperatorDef was never set.  Use Caffe2Annotation::setOperatorDef.");
    return OpDef;
  }

  static bool classof(const Annotation *A) {
    return A->getKind() == AnnotationKind::Caffe2;
  }

private:
  std::string Device = "";
  caffe2::OperatorDef* OpDef = nullptr;
};

nom::repr::NNModule convertToNNModule(caffe2::NetDef &net, std::unordered_map<std::string, nom::repr::NNGraph::NodeRef>* blobMapOut = nullptr);

caffe2::NetDef convertToCaffe2Proto(nom::repr::NNModule&);

// Pass in an oldNet to copy all the attributes of that network.
// Be warned that transformations that modify the graph's inputs or outputs
// are not reflected in changes to external_input or external_output.
caffe2::NetDef convertToCaffe2Proto(nom::repr::NNModule&, const caffe2::NetDef& oldNet);

std::unique_ptr<nom::repr::NeuralNetOperator> convertToOperatorDef(caffe2::OperatorDef op);

} // namespace caffe2 


#endif // NOM_CONVERTERS_CAFFE2_H
