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
    std::vector<NNGraph::NodeRef> ns;
    auto nn = caffe2::convertToNNModule(proto, false, &ns);
    return std::pair<NNModule, std::vector<NNGraph::NodeRef>>(
        std::move(nn), ns);
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

  m.def("replaceProducer", &nn::replaceProducer);
  m.def("replaceAllUsesWith", &nn::replaceAllUsesWith);
  m.def("replaceAsConsumer", &nn::replaceAsConsumer);

  py::class_<NNModule> nnmodule(m, "NNModule");
  nnmodule.def(py::init<>())
      .def(
          "dataFlow",
          [](NNModule* nn) -> NNGraph* { return &nn->dataFlow; },
          py::return_value_policy::reference_internal)
      .def(
          "createUniqueDataNode",
          &NNModule::createUniqueDataNode,
          py::return_value_policy::reference_internal)
      .def(
          "convertToCaffe2Proto",
          [](NNModule& nn, py::object def) {
            CAFFE_ENFORCE(
                pybind11::hasattr(def, "SerializeToString"),
                "convertToCaffe2Proto takes either no args",
                "a NetDef");
            auto str = def.attr("SerializeToString")();
            caffe2::NetDef proto;
            proto.ParseFromString(py::bytes(str));
            auto new_proto = caffe2::convertToCaffe2Proto(nn, proto);
            std::string out;
            new_proto.SerializeToString(&out);
            return py::bytes(out);
          })
      .def(
          "getExecutionOrder",
          [](NNModule& nn) {
            nn::coalesceInsertedDataDependencies(&nn);
            std::vector<NNGraph::NodeRef> out;
            auto sccs = nom::algorithm::tarjans(&nn.controlFlow);
            for (const auto& scc : sccs) {
              for (const auto& bb : scc.getNodes()) {
                for (const auto& instr : bb->data().getInstructions()) {
                  out.emplace_back(instr);
                }
              }
            }
            return out;
          },
          py::return_value_policy::reference_internal)
      .def("replaceSubgraph", &NNModule::replaceSubgraph)
      .def("deleteSubgraph", &NNModule::deleteSubgraph);

  auto getTensors = [](NNGraph* g) {
    return nn::nodeIterator<nom::repr::Tensor>(*g);
  };
  auto getOperators = [](NNGraph* g) {
    return nn::nodeIterator<NeuralNetOperator>(*g);
  };
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
      .def("deleteEdge", &NNGraph::deleteEdge)
      .def(
          "deleteEdge",
          [](NNGraph* g, NNGraph::NodeRef a, NNGraph::NodeRef b) {
            auto edge = g->getEdgeIfExists(a, b);
            if (edge) {
              g->deleteEdge(edge);
            }
          })
      .def(
          "createNode",
          [](NNGraph* g, GenericOperator& op) {
            return g->createNode(
                std::make_unique<GenericOperator>(op.getName()));
          },
          py::return_value_policy::reference_internal)
      .def(
          "createNode",
          [](NNGraph* g, nom::repr::Tensor& tensor) {
            return g->createNode(
                std::make_unique<nom::repr::Tensor>(tensor.getName()));
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
      .def("deleteNode", &NNGraph::deleteNode)
      .def(
          "replaceNode",
          [](NNGraph* g, NNGraph::NodeRef old_node, NNGraph::NodeRef new_node) {
            g->replaceNode(old_node, new_node);
          })
      .def(
          "getMutableNodes",
          &NNGraph::getMutableNodes,
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "nodes",
          &NNGraph::getMutableNodes,
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "operators",
          getOperators,
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "tensors", getTensors, py::return_value_policy::reference_internal);

  // Node level methods
  using NodeType = nom::Node<std::unique_ptr<nom::repr::Value>>;
  py::class_<NodeType> noderef(m, "NodeRef");
  auto getName = [](NNGraph::NodeRef n) {
    if (nn::is<nom::repr::Tensor>(n)) {
      return nn::get<nom::repr::Tensor>(n)->getName();
    } else if (nn::is<NeuralNetOperator>(n)) {
      return nn::get<NeuralNetOperator>(n)->getName();
    }
    return std::string("Unknown");
  };
  auto getType = [](NNGraph::NodeRef n) {
    if (nn::is<nom::repr::Tensor>(n)) {
      return "Tensor";
    } else if (nn::is<NeuralNetOperator>(n)) {
      return "Operator";
    }
    return "Unknown";
  };
  auto getOperator = [](NNGraph::NodeRef n) {
    CAFFE_ENFORCE(nn::is<NeuralNetOperator>(n));
    return nn::get<NeuralNetOperator>(n);
  };
  auto getTensor = [](NNGraph::NodeRef n) {
    CAFFE_ENFORCE(nn::is<nom::repr::Tensor>(n));
    return nn::get<nom::repr::Tensor>(n);
  };
  auto getInputs = [](NNGraph::NodeRef n) {
    CAFFE_ENFORCE(nn::is<NeuralNetOperator>(n));
    return nn::getInputs(n);
  };
  auto getOutputs = [](NNGraph::NodeRef n) {
    CAFFE_ENFORCE(nn::is<NeuralNetOperator>(n));
    return nn::getOutputs(n);
  };
  auto getProducer = [](NNGraph::NodeRef n) {
    CAFFE_ENFORCE(nn::is<NeuralNetData>(n));
    return nn::getProducer(n);
  };
  auto getConsumers = [](NNGraph::NodeRef n) {
    CAFFE_ENFORCE(nn::is<NeuralNetData>(n));
    return nn::getConsumers(n);
  };
  auto setAnnotation = [](NNGraph::NodeRef n, Caffe2Annotation& annot) {
    auto* nnOp = nn::get<NeuralNetOperator>(n);
    nnOp->setAnnotation(std::make_unique<Caffe2Annotation>(annot));
  };
  auto getAnnotation = [](NNGraph::NodeRef n) {
    return getOrAddCaffe2Annotation(n);
  };

  noderef
      .def(
          "isOperator",
          [](NNGraph::NodeRef n) { return nn::is<NeuralNetOperator>(n); })
      .def(
          "isTensor",
          [](NNGraph::NodeRef n) { return nn::is<nom::repr::Tensor>(n); })
      .def("getType", getType)
      .def_property_readonly("type", getType)
      .def("getName", getName)
      .def_property_readonly("name", getName)
      .def(
          "getOperator",
          getOperator,
          py::return_value_policy::reference_internal)
      .def("getTensor", getTensor, py::return_value_policy::reference_internal)
      .def_property_readonly(
          "operator", getOperator, py::return_value_policy::reference)
      .def_property_readonly(
          "tensor", getTensor, py::return_value_policy::reference)
      .def("getInputs", getInputs, py::return_value_policy::reference)
      .def("getOutputs", getOutputs, py::return_value_policy::reference)
      .def("hasProducer", [](NNGraph::NodeRef n) { return nn::hasProducer(n); })
      .def("getProducer", getProducer, py::return_value_policy::reference)
      .def("getConsumers", getConsumers, py::return_value_policy::reference)
      .def_property_readonly(
          "inputs", getInputs, py::return_value_policy::reference)
      .def_property_readonly(
          "outputs", getOutputs, py::return_value_policy::reference)
      .def_property_readonly(
          "producer", getProducer, py::return_value_policy::reference)
      .def_property_readonly(
          "consumers", getConsumers, py::return_value_policy::reference)
      .def("getAnnotation", getAnnotation, py::return_value_policy::reference)
      .def("setAnnotation", setAnnotation)
      .def_property(
          "annotation",
          getAnnotation,
          setAnnotation,
          py::return_value_policy::reference)
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
  nnsubgraph.def(py::init<>())
      .def("__len__", [](NNSubgraph& s) { return s.getNodes().size(); })
      .def(
          "__repr__",
          [](NNSubgraph* g) {
            return nom::converters::convertToDotString<NNGraph>(*g, NNPrinter);
          })
      .def(
          "addNode",
          [](NNSubgraph* sg, NNGraph::NodeRef node) { sg->addNode(node); })
      .def(
          "induceEdges",
          [](NNSubgraph* sg) { nom::algorithm::induceEdges(sg); })
      .def_property_readonly(
          "nodes",
          [](NNSubgraph& s) {
            std::vector<NNGraph::NodeRef> out;
            for (auto n : s.getNodes()) {
              out.emplace_back(n);
            }
            return out;
          },
          py::return_value_policy::reference)
      .def("hasNode", [](NNSubgraph& s, NNGraph::NodeRef n) {
        return s.hasNode(n);
      });

  py::class_<nn::NNMatchGraph> nnMatchGraph(m, "NNMatchGraph");
  nnMatchGraph.def(py::init<>());

  using MatchPredicateType = nom::Node<nn::NNMatchPredicate>;
  py::class_<MatchPredicateType> nnMatchPredicate(m, "MatchPredicateRef");

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
            auto match = [opName](NNGraph::NodeRef node) {
              NOM_REQUIRE_OR_RET_FALSE(nn::is<NeuralNetOperator>(node));
              auto nnOp = nn::get<NeuralNetOperator>(node);
              return opName == nnOp->getName();
            };
            auto node = nn::NNMatchPredicate(match);
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
            auto node = nn::NNMatchPredicate(nn::is<nom::repr::Tensor>);
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
            auto match = [](NNGraph::NodeRef node) { return true; };
            auto node = nn::NNMatchPredicate(match);
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
    auto result = mg->isSubgraphMatch(node, match_node, false);
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
      .def("getComponentLevels", &Caffe2Annotation::getComponentLevels)
      .def("hasDeviceOption", &Caffe2Annotation::hasDeviceOption)
      .def_property(
          "device_option",
          [](Caffe2Annotation& annot) {
            auto DeviceOption = py::module::import("caffe2.proto.caffe2_pb2")
                                    .attr("DeviceOption");
            // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
            auto proto = annot.getDeviceOption();
            std::string serialized_proto;
            proto.SerializeToString(&serialized_proto);
            auto py_device_opt = DeviceOption();
            py_device_opt.attr("ParseFromString")(py::bytes(serialized_proto));
            return py_device_opt;
          },
          [](Caffe2Annotation& annot, py::object& def) {
            CAFFE_ENFORCE(
                pybind11::hasattr(def, "SerializeToString"),
                "device_option can only be set to a DeviceOption");
            auto str = def.attr("SerializeToString")();
            caffe2::DeviceOption proto;
            proto.ParseFromString(py::bytes(str));
            annot.setDeviceOption(proto);
          },
          py::return_value_policy::reference)
      .def_property(
          "operator_def",
          [](Caffe2Annotation& annot) {
            auto opDef = py::module::import("caffe2.proto.caffe2_pb2")
                                    .attr("OperatorDef");
            // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
            auto proto = annot.getOperatorDef();
            std::string serialized_proto;
            proto.SerializeToString(&serialized_proto);
            auto py_op_def= opDef();
            py_op_def.attr("ParseFromString")(py::bytes(serialized_proto));
            return py_op_def;
          },
          [](Caffe2Annotation& annot, py::object& def) {
            CAFFE_ENFORCE(
                pybind11::hasattr(def, "SerializeToString"),
                "operator_def can only be set to an OperatorDef");
            auto str = def.attr("SerializeToString")();
            caffe2::OperatorDef proto;
            proto.ParseFromString(py::bytes(str));
            annot.setOperatorDef(proto);
          },
          py::return_value_policy::reference);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_PYBIND_ADDITION(addNomnigraphMethods);

} // namespace python
} // namespace caffe2
