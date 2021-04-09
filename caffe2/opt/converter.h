#ifndef CAFFE2_OPT_CONVERTER_H
#define CAFFE2_OPT_CONVERTER_H

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/opt/annotations.h"
#include "caffe2/proto/caffe2_pb.h"
#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Representations/ControlFlow.h"
#include "nomnigraph/Representations/NeuralNet.h"

#include <unordered_map>

namespace caffe2 {

TORCH_API void injectDataEdgeIndicators(caffe2::NetDef* net);
TORCH_API void removeDataEdgeIndicators(caffe2::NetDef* net);

// Default conversion to a NNModule
// Optionally strict -- which checks for various input and output conditions.
// Optionally this function will update a vector that maps operators in the
// netdef positionally to NodeRefs in the resultant NNModule.
TORCH_API nom::repr::NNModule convertToNNModule(
    const caffe2::NetDef& net,
    bool strict = false,
    std::vector<nom::repr::NNGraph::NodeRef>* = nullptr);
TORCH_API caffe2::NetDef convertToCaffe2Proto(nom::repr::NNModule&);

// Pass in an oldNet to copy all the attributes of that network.
// Be warned that transformations that modify the graph's inputs or outputs
// are not reflected in changes to external_input or external_output.
TORCH_API caffe2::NetDef convertToCaffe2Proto(
    nom::repr::NNModule&,
    const caffe2::NetDef& oldNet);

// Use these functions instead of the registry directly.
TORCH_API std::unique_ptr<nom::repr::NeuralNetOperator>
convertToNeuralNetOperator(const caffe2::OperatorDef& op);

TORCH_API caffe2::OperatorDef convertToOperatorDef(
    const nom::repr::NNGraph::NodeRef& instrNode);

// If the annotation doesn't exist, attempt to add it
TORCH_API Caffe2Annotation* getOrAddCaffe2Annotation(
    nom::repr::NNGraph::NodeRef& instrNode);

class TORCH_API Converter {
 public:
  explicit Converter() = default;
  virtual std::unique_ptr<nom::repr::NeuralNetOperator>
  convertToNeuralNetOperator(const OperatorDef&) = 0;
  virtual OperatorDef convertToOperatorDef(const nom::repr::NeuralNetOperator*);
  static std::map<std::string, caffe2::Argument> getArgumentsFromOperator(
      caffe2::OperatorDef op);

  virtual ~Converter() {}

 protected:
  caffe2::DeviceOption getDeviceOption(
      const nom::repr::NeuralNetOperator* nnOp) const;
};

C10_DECLARE_REGISTRY(ConverterRegistry, Converter);
#define REGISTER_CONVERTER(name, cls) \
  C10_REGISTER_CLASS(ConverterRegistry, name, cls)

#define TRIVIAL_CONVERTER(opName)                                             \
  class opName##Converter : public Converter {                                \
    std::unique_ptr<nom::repr::NeuralNetOperator> convertToNeuralNetOperator( \
        const OperatorDef& op) override {                                     \
      return std::make_unique<nom::repr::opName>();                     \
    }                                                                         \
    virtual ~opName##Converter() {}                                           \
  };

} // namespace caffe2

#endif // CAFFE2_OPT_CONVERTER_H
