#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"
#include "caffe2/opt/converter.h"
#include "caffe2/opt/distributed.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/python/dlpack.h"
#include "caffe2/python/pybind_state_registry.h"
#include "caffe2/utils/proto_utils.h"
#include "nomnigraph/Converters/Dot.h"
#include "nomnigraph/Graph/Algorithms.h"
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

using Graph = nom::Graph<py::object>;
std::map<std::string, std::string> GraphPrinter(typename Graph::NodeRef node) {
  std::map<std::string, std::string> labelMap;
  assert(node->data() && "Node doesn't have data, can't render it");
  labelMap["label"] = py::str(node->data());
  return labelMap;
};

} // namespace

void addNomnigraphMethods(pybind11::module& m) {
  // Generic Graph methods
  py::class_<Graph> graph(m, "Graph");
  py::class_<nom::Node<py::object>> node(m, "Node");
  py::class_<nom::Edge<py::object>> edge(m, "Edge");
  graph.def(py::init<>())
      .def(
          "__repr__",
          [](Graph* g) {
            return nom::converters::convertToDotString(g, GraphPrinter);
          })
      .def(
          "createEdge",
          [](Graph* g, Graph::NodeRef a, Graph::NodeRef b) {
            return g->createEdge(a, b);
          },
          py::return_value_policy::reference_internal)
      .def(
          "createNode",
          [](Graph* g, py::object obj) {
            return g->createNode(std::move(obj));
          },
          py::return_value_policy::reference_internal);

  // NNModule methods
  m.def("NNModuleFromProtobuf", [](py::bytes def) {
    caffe2::NetDef proto;
    CAFFE_ENFORCE(ParseProtoFromLargeString(def.cast<std::string>(), &proto));
    return caffe2::convertToNNModule(proto);
  });

  m.def(
      "NNModuleFromProtobufDistributed",
      [](py::bytes def, std::map<std::string, py::bytes> blobToDeviceMap) {
        std::map<std::string, caffe2::DeviceOption> m;
        for (const auto& el : blobToDeviceMap) {
          caffe2::DeviceOption d;
          CAFFE_ENFORCE(
              ParseProtoFromLargeString(el.second.cast<std::string>(), &d));
          m[el.first] = d;
        }

        caffe2::NetDef proto;
        CAFFE_ENFORCE(
            ParseProtoFromLargeString(def.cast<std::string>(), &proto));

        return caffe2::convertToNNModule(proto, m);
      });

  py::class_<NNModule> nnmodule(m, "NNModule");
  nnmodule.def(py::init<>())
      .def(
          "dataFlow",
          [](NNModule* nn) -> NNGraph* { return &nn->dataFlow; },
          py::return_value_policy::reference_internal)
      .def("convertToCaffe2Proto", [](NNModule& nn, py::object def) {
        CAFFE_ENFORCE(
            pybind11::hasattr(def, "SerializeToString"),
            "convertToCaffe2Proto takes either no args", "a NetDef");
        auto str = def.attr("SerializeToString")();
        caffe2::NetDef proto;
        proto.ParseFromString(py::bytes(str));
        auto new_proto = caffe2::convertToCaffe2Proto(nn, proto);
        std::string out;
        new_proto.SerializeToString(&out);
        return py::bytes(out);
      });

  // NNGraph methods
  py::class_<NNGraph> nngraph(m, "NNGraph");
  nngraph
      .def(
          "__repr__",
          [](NNGraph* g) {
            return nom::converters::convertToDotString(g, NNPrinter);
          })
      .def(
          "createEdge",
          [](NNGraph* g, NNGraph::NodeRef a, NNGraph::NodeRef b) {
            CAFFE_ENFORCE(
                (nn::is<NeuralNetOperator>(a) && nn::is<NeuralNetData>(b)) ||
                    (nn::is<NeuralNetOperator>(b) && nn::is<NeuralNetData>(a)),
                "Edges must exist between NeuralNetOperator and NeuralNetData");
            g->createEdge(a, b);
          })

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
          "createNode",
          [](NNGraph* g, py::object op_def) {
            CAFFE_ENFORCE(
                pybind11::hasattr(op_def, "SerializeToString"),
                "createNode takes either OperatorDef",
                "or ng.NeuralNetOperator");
            auto str = op_def.attr("SerializeToString")();
            OperatorDef op;
            op.ParseFromString(py::bytes(str));
            if (op.input().size() || op.output().size()) {
              LOG(WARNING)
                  << "Input and output specifications are "
                  << "dropped when converting a single operator to nomnigraph. "
                  << "Use ng.NNModule(NetDef&) to preserve these.";
            }
            return g->createNode(convertToNeuralNetOperator(op));
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
          py::return_value_policy::reference_internal)
      .def_property(
          "annotation",
          [](NNGraph::NodeRef n) { return getOrAddCaffe2Annotation(n); },
          [](NNGraph::NodeRef n, Caffe2Annotation annot) {
            auto* nnOp = nn::get<NeuralNetOperator>(n);
            nnOp->setAnnotation(
                nom::util::make_unique<Caffe2Annotation>(annot));
          },
          py::return_value_policy::copy)
      .def(
          "getAnnotation",
          [](NNGraph::NodeRef n) { return getOrAddCaffe2Annotation(n); },
          py::return_value_policy::copy)
      .def(
          "setAnnotation",
          [](NNGraph::NodeRef n, Caffe2Annotation annot) {
            auto* nnOp = nn::get<NeuralNetOperator>(n);
            nnOp->setAnnotation(
                nom::util::make_unique<Caffe2Annotation>(annot));
          })
      .def(
          "getOperatorPredecessors",
          [](NNGraph::NodeRef n) {
            CAFFE_ENFORCE(nn::is<NeuralNetOperator>(n));
            std::vector<NNGraph::NodeRef> pred;
            for (const auto& inEdge : n->getInEdges()) {
              auto data = inEdge->tail();
              if (nn::hasProducer(data)) {
                pred.emplace_back(nn::getProducer(data));
              }
            }
            return pred;
          },
          py::return_value_policy::reference)
      .def(
          "getOperatorSuccessors",
          [](NNGraph::NodeRef n) {
            CAFFE_ENFORCE(nn::is<NeuralNetOperator>(n));
            std::vector<NNGraph::NodeRef> succ;
            for (const auto& outEdge : n->getOutEdges()) {
              auto data = outEdge->head();
              for (const auto& consumer : nn::getConsumers(data)) {
                succ.emplace_back(consumer);
              }
            }
            return succ;
          },
          py::return_value_policy::reference);

  py::class_<NeuralNetOperator, GenericOperator> nnop(m, "NeuralNetOperator");
  py::class_<nom::repr::Tensor> nndata(m, "NeuralNetData");

  nnop.def(py::init<std::string>()).def("getName", &NeuralNetOperator::getName);
  nndata.def(py::init<std::string>()).def("getName", &NeuralNetData::getName);

  // Subgraph matching API
  py::class_<NNSubgraph> nnsubgraph(m, "NNSubgraph");
  nnsubgraph.def("__len__", [](NNSubgraph& s) { return s.getNodes().size(); });

  py::class_<nn::NNMatchGraph> nnMatchGraph(m, "NNMatchGraph");
  nnMatchGraph.def(py::init<>());

  using MatchNodeType =
      nom::Node<nom::matcher::MatchNode<nn::NNNodeMatchCriteria>>;
  py::class_<MatchNodeType> nnMatchNode(m, "MatchNodeRef");

  nnMatchGraph
      .def(
          "createEdge",
          [](nn::NNMatchGraph* g,
             nn::NNMatchGraph::NodeRef a,
             nn::NNMatchGraph::NodeRef b) { g->createEdge(a, b); })
      .def(
          "createNode",
          [](nn::NNMatchGraph* g, GenericOperator& op, bool strict) {
            auto opName = op.getName();
            auto match =
                nn::NNNodeMatchCriteria([opName](NNGraph::NodeRef node) {
                  NOM_REQUIRE_OR_RET_FALSE(nn::is<NeuralNetOperator>(node));
                  auto nnOp = nn::get<NeuralNetOperator>(node);
                  return opName == nnOp->getName();
                });
            auto node = nom::matcher::MatchNode<nn::NNNodeMatchCriteria>(match);
            if (!strict) {
              node.nonTerminal();
            }
            return g->createNode(std::move(node));
          },
          py::return_value_policy::reference_internal,
          py::arg("node"),
          py::arg("strict") = false)
      .def(
          "createNode",
          [](nn::NNMatchGraph* g, nom::repr::Tensor& tensor, bool strict) {
            auto node = nn::NNMatchNode(nn::matchTensor());
            if (!strict) {
              node.nonTerminal();
            }
            return g->createNode(std::move(node));
          },
          py::return_value_policy::reference_internal,
          py::arg("tensor"),
          py::arg("strict") = false)
      .def(
          "createNode",
          [](nn::NNMatchGraph* g, bool strict) {
            auto match = nn::NNNodeMatchCriteria(
                [](NNGraph::NodeRef node) { return true; });
            auto node = nom::matcher::MatchNode<nn::NNNodeMatchCriteria>(match);
            if (!strict) {
              node.nonTerminal();
            }
            return g->createNode(std::move(node));
          },
          py::return_value_policy::reference_internal,
          py::arg("strict") = false)
      .def(
          "getMutableNodes",
          [](nn::NNMatchGraph* g) { return g->getMutableNodes(); },
          py::return_value_policy::reference_internal);

  m.def("matchSubgraph", [](NNGraph::NodeRef node, nn::NNMatchGraph* mg) {
    // Get root node or node in root cycle
    auto match_node = *nom::algorithm::tarjans(mg).back().getNodes().begin();
    auto result =
        nn::NNSubgraphMatcher::isSubgraphMatch(node, match_node, false);
    if (result.isMatch()) {
      return *result.getMatchedSubgraph();
    }
    return NNSubgraph();
  });

  // Annotation API
  py::class_<Caffe2Annotation> annotation(m, "Annotation");
  annotation.def(py::init<>())
      .def("setDevice", &Caffe2Annotation::setDevice)
      .def("getDevice", &Caffe2Annotation::getDevice)
      .def("setDeviceType", &Caffe2Annotation::setDeviceType)
      .def("getDeviceType", &Caffe2Annotation::getDeviceType)
      .def("setKeyNode", &Caffe2Annotation::setKeyNode)
      .def(
          "getKeyNode",
          &Caffe2Annotation::getKeyNode,
          py::return_value_policy::reference)
      .def("setLengthNode", &Caffe2Annotation::setLengthNode)
      .def(
          "getLengthNode",
          &Caffe2Annotation::getLengthNode,
          py::return_value_policy::reference)
      .def("setComponentLevels", &Caffe2Annotation::setComponentLevels)
      .def("getComponentLevels", &Caffe2Annotation::getComponentLevels);
}

REGISTER_PYBIND_ADDITION(addNomnigraphMethods);

} // namespace python
} // namespace caffe2
