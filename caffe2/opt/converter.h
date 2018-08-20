#ifndef CAFFE2_OPT_CONVERTER_H
#define CAFFE2_OPT_CONVERTER_H

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/proto/caffe2.pb.h"
#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Representations/ControlFlow.h"
#include "nomnigraph/Representations/NeuralNet.h"

#include <unordered_map>

namespace caffe2 {

class CAFFE2_API Caffe2Annotation : public nom::repr::Annotation {
public:
  Caffe2Annotation() : Annotation(AnnotationKind::Caffe2) {}
  Caffe2Annotation(std::string device)
      : Annotation(AnnotationKind::Caffe2), Device(device) {}
  virtual ~Caffe2Annotation() {}

  void setDevice(std::string device) { Device = device; }
  const std::string getDevice() const { return Device; }

  void setDeviceType(int device) {
    DeviceType = device;
  }
  int getDeviceType() const {
    return DeviceType;
  }

  void setOperatorDef(const caffe2::OperatorDef& opDef) {
    OpDef = opDef;
    OpDefExists = true;
  }
  const caffe2::OperatorDef& getOperatorDef() const {
    CAFFE_ENFORCE(
        OpDefExists,
        "OperatorDef was never set.  Use Caffe2Annotation::setOperatorDef.");
    return OpDef;
  }
  caffe2::OperatorDef* getMutableOperatorDef() {
    CAFFE_ENFORCE(
        OpDefExists,
        "OperatorDef was never set.  Use Caffe2Annotation::setOperatorDef.");
    return &OpDef;
  }

  static bool classof(const Annotation *A) {
    return A->getKind() == AnnotationKind::Caffe2;
  }

private:
  std::string Device = "";
  caffe2::OperatorDef OpDef;
  bool OpDefExists = false;
  int DeviceType = caffe2::DeviceType::CPU;
};

CAFFE2_API nom::repr::NNModule convertToNNModule(caffe2::NetDef &net, std::unordered_map<std::string, nom::repr::NNGraph::NodeRef>* blobMapOut = nullptr);

CAFFE2_API caffe2::NetDef convertToCaffe2Proto(nom::repr::NNModule&);

// Pass in an oldNet to copy all the attributes of that network.
// Be warned that transformations that modify the graph's inputs or outputs
// are not reflected in changes to external_input or external_output.
CAFFE2_API caffe2::NetDef convertToCaffe2Proto(nom::repr::NNModule&, const caffe2::NetDef& oldNet);

// Use these functions instead of the registry directly.
CAFFE2_API std::unique_ptr<nom::repr::NeuralNetOperator> convertToNeuralNetOperator(
    const caffe2::OperatorDef& op);

CAFFE2_API caffe2::OperatorDef convertToOperatorDef(
    const nom::repr::NNGraph::NodeRef& instrNode);

class CAFFE2_API Converter {
 public:
  explicit Converter() {}
  virtual std::unique_ptr<nom::repr::NeuralNetOperator>
  convertToNeuralNetOperator(const OperatorDef&) = 0;
  virtual OperatorDef convertToOperatorDef(const nom::repr::NeuralNetOperator*);
  static std::map<std::string, caffe2::Argument> getArgumentsFromOperator(
      caffe2::OperatorDef op);

  virtual ~Converter() {}
};

CAFFE_DECLARE_REGISTRY(ConverterRegistry, Converter);
#define REGISTER_CONVERTER(name, cls) \
  CAFFE_REGISTER_CLASS(ConverterRegistry, name, cls)

#define TRIVIAL_CONVERTER(opName)                                             \
  class opName##Converter : public Converter {                                \
    std::unique_ptr<nom::repr::NeuralNetOperator> convertToNeuralNetOperator( \
        const OperatorDef& op) override {                                     \
      return util::make_unique<repr::opName>();                               \
    }                                                                         \
    virtual ~opName##Converter() {}                                           \
  };

} // namespace caffe2


#endif // CAFFE2_OPT_CONVERTER_H
