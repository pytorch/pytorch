#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"
#include "caffe2/opt/converter.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/python/dlpack.h"
#include "caffe2/python/pybind_state_registry.h"
#include "caffe2/utils/proto_utils.h"
#include "nomnigraph/Converters/Dot.h"
#include "nomnigraph/Representations/NeuralNet.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using ListCasterBase = pybind11::detail::list_caster<
    std::vector<nom::repr::NNGraph::NodeRef>,
    nom::repr::NNGraph::NodeRef>;
namespace pybind11 {
namespace detail {
template <>
struct type_caster<std::vector<nom::repr::NNGraph::NodeRef>> : ListCasterBase {
  static handle cast(
      const std::vector<nom::repr::NNGraph::NodeRef>& src,
      return_value_policy,
      handle parent) {
    return ListCasterBase::cast(src, return_value_policy::reference, parent);
  }
  static handle cast(
      const std::vector<nom::repr::NNGraph::NodeRef>* src,
      return_value_policy pol,
      handle parent) {
    return cast(*src, pol, parent);
  }
};
} // namespace detail
} // namespace pybind11

namespace caffe2 {
namespace python {

using namespace nom::repr;

namespace {

std::map<std::string, std::string> NNPrinter(
    typename nom::repr::NNGraph::NodeRef node) {
  std::map<std::string, std::string> labelMap;
  assert(node->data() && "Node doesn't have data, can't render it");
  if (isa<nom::repr::NeuralNetOperator>(node->data())) {
    auto* op = dyn_cast<nom::repr::NeuralNetOperator>(node->data().get());
    labelMap["label"] = op->getName();
    labelMap["shape"] = "box";
  } else if (isa<nom::repr::Data>(node->data())) {
    auto tensor = dyn_cast<nom::repr::NeuralNetData>(node->data().get());
    labelMap["label"] = tensor->getName();
  }
  return labelMap;
};

} // namespace

void addNomnigraphMethods(pybind11::module& m) {
  LOG(ERROR) << "I'm here";
  py::class_<NNModule> nnmodule(m, "NNModule");

  m.def("NNModuleFromProtobuf", [](py::bytes def) {
    caffe2::NetDef proto;
    CAFFE_ENFORCE(ParseProtoFromLargeString(def.cast<std::string>(), &proto));
    return caffe2::convertToNNModule(proto);
  }); //, py::return_value_policy::reference_internal);

  // NNModule methods
  nnmodule.def(py::init<>())
      .def(
          "dataFlow",
          [](NNModule* nn) -> NNGraph* { return &nn->dataFlow; },
          py::return_value_policy::reference_internal)
      .def("dotString", [](NNModule* nn) {
        auto str =
            nom::converters::convertToDotString(&nn->dataFlow, NNPrinter);
        return str;
      });

  // NNGraph methods
  py::class_<NNGraph> nngraph(m, "NNGraph");

  nngraph.def(
      "createEdge", [](NNGraph* g, NNGraph::NodeRef a, NNGraph::NodeRef b) {
        CAFFE_ENFORCE(
            (nn::is<NeuralNetOperator>(a) && nn::is<NeuralNetData>(b)) ||
                (nn::is<NeuralNetOperator>(b) && nn::is<NeuralNetData>(a)),
            "Edges must exist between NeuralNetOperator and NeuralNetData");
        g->createEdge(a, b);
      });

  nngraph
      .def(
          "createNode",
          [](NNGraph* g, GenericOperator& op) {
            return g->createNode(
                nom::util::make_unique<GenericOperator>(op.getName()));
          },
          py::return_value_policy::reference_internal)
      .def(
          "createNode",
          [](NNGraph* g, nom::repr::Tensor& tensor) {
            return g->createNode(
                nom::util::make_unique<nom::repr::Tensor>(tensor.getName()));
          },
          py::return_value_policy::reference_internal)
      .def(
          "getMutableNodes",
          [](NNGraph* g) { return g->getMutableNodes(); },
          py::return_value_policy::reference_internal);

  // Node level methods
  using NodeType = nom::Node<std::unique_ptr<nom::repr::Value>>;
  py::class_<NodeType> noderef(m, "NodeRef");

  noderef
      .def(
          "isOperator",
          [](NNGraph::NodeRef n) { return nn::is<NeuralNetOperator>(n); })
      .def(
          "isTensor",
          [](NNGraph::NodeRef n) { return nn::is<nom::repr::Tensor>(n); })
      .def(
          "getOperator",
          [](NNGraph::NodeRef n) {
            CAFFE_ENFORCE(nn::is<NeuralNetOperator>(n));
            return nn::get<NeuralNetOperator>(n);
          },
          py::return_value_policy::reference_internal)
      .def(
          "getTensor",
          [](NNGraph::NodeRef n) {
            CAFFE_ENFORCE(nn::is<nom::repr::Tensor>(n));
            return nn::get<nom::repr::Tensor>(n);
          },
          py::return_value_policy::reference_internal);

  py::class_<GenericOperator> nnop(m, "NeuralNetOperator");
  py::class_<nom::repr::Tensor> nndata(m, "NeuralNetData");

  nnop.def(py::init<std::string>()).def("getName", &NeuralNetOperator::getName);
  nndata.def(py::init<std::string>()).def("getName", &NeuralNetData::getName);
}

REGISTER_PYBIND_ADDITION(addNomnigraphMethods);

} // namespace python
} // namespace caffe2
