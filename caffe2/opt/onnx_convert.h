#include "caffe2/core/common.h"

class TORCH_API OnnxAnnotation : public nom::repr::Annotation {
public:
  OnnxAnnotation() : Annotation(AnnotationKind::Onnx) {}
  OnnxAnnotation(std::string device)
      : Annotation(AnnotationKind::Onnx), Device(device) {}

  void setDevice(std::string device) { Device = device; }
  const std::string getDevice() const { return Device; }

  void setOperatorDef(caffe2::OperatorDef* opDef) {
    OpDef = opDef;
  }
  const caffe2::OperatorDef* getOperatorDef() const {
    assert(OpDef && "OperatorDef was never set.  Use OnnxAnnotation::setOperatorDef.");
    return OpDef;
  }
  caffe2::OperatorDef* getMutableOperatorDef() {
    assert(OpDef && "OperatorDef was never set.  Use OnnxAnnotation::setOperatorDef.");
    return OpDef;
  }

  static bool classof(const Annotation *A) {
    return A->getKind() == AnnotationKind::Onnx;
  }

private:
  std::string Device = "";
  caffe2::OperatorDef* OpDef = nullptr;
};

TORCH_API nom::repr::NNModule convertToNNModule(caffe2::NetDef &net, std::unordered_map<std::string, nom::repr::NNGraph::NodeRef>* blobMapOut = nullptr);

TORCH_API caffe2::NetDef convertToOnnxProto(nom::repr::NNModule&);

TORCH_API std::unique_ptr<nom::repr::NeuralNetOperator> convertToOperatorDef(caffe2::OperatorDef op);
