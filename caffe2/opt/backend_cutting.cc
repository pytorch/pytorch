#include "caffe2/opt/backend_cutting.h"
#include "caffe2/opt/converter.h"
#include "caffe2/core/logging.h"
#include "nomnigraph/Converters/Dot.h"
#include "nomnigraph/Representations/NeuralNet.h"

#include <algorithm>
#include <fstream>
#include <queue>

namespace caffe2 {
namespace opt {

namespace {

using namespace nom::repr;
using NodeRef = NNGraph::NodeRef;
using EdgeRef = NNGraph::EdgeRef;

class GroupAnnotation {
 public:
  GroupAnnotation(int i, int g = -1) : group(g), in_degree(i) {}
  int group;
  int in_degree;
  bool needs_transform{true};
};

struct VisitorContext {
  VisitorContext(std::function<bool(const caffe2::OperatorDef&)> func)
      : predicate(func) {}
  std::unordered_map<NodeRef, GroupAnnotation> infos;
  std::unordered_set<NodeRef> frontier;
  std::vector<NodeRef> current_group;
  std::function<bool(const caffe2::OperatorDef&)> predicate;

  int group{0};
  bool find_supported{true};
};

std::string ShowNode(NodeRef node) {
  if (nn::is<NeuralNetData>(node)) {
    const auto* nn_tensor = nn::get<NeuralNetData>(node);
    return MakeString("Tensor: ", nn_tensor->getName());
  } else if (nn::is<NeuralNetOperator>(node)) {
    const auto* nn_op = nn::get<NeuralNetOperator>(node);
    const auto* op_def = dyn_cast<Caffe2Annotation>(nn_op->getAnnotation())->getOperatorDef();
    CAFFE_ENFORCE(op_def);
    return MakeString("Op: ", op_def->type());
  } else {
    CAFFE_THROW("Known node");
  }
}

void DumpGraph(NNGraph* g) {
  auto nnprinter = [](typename NNGraph::NodeRef node) {
    std::map<std::string, std::string> labelMap;
    assert(node->data() && "Node doesn't have data, can't render it");
    if (isa<NeuralNetOperator>(node->data())) {
      auto* op = dyn_cast<NeuralNetOperator>(node->data().get());
      labelMap["label"] =
          op->getName() + " (" + caffe2::to_string((unsigned long long)node) + ")";
      auto* annotation = op->getAnnotation();
      if (annotation && isa<Caffe2Annotation>(annotation)) {
        auto device_annotation =
            dyn_cast<Caffe2Annotation>(annotation);
        labelMap["label"] += "\\n[" + device_annotation->getDevice() + "]";
        auto hash = std::hash<std::string>{}(device_annotation->getDevice());
        std::stringstream hex_stream;
        hex_stream << std::hex << hash;
        labelMap["color"] = "#" + hex_stream.str().substr(0, 6);
        labelMap["fontcolor"] = labelMap["color"];
      }
      labelMap["shape"] = "box";
    } else if (isa<Data>(node->data())) {
      auto tensor = dyn_cast<NeuralNetData>(node->data().get());
      labelMap["label"] = tensor->getName();
      labelMap["label"] += "_" + caffe2::to_string(tensor->getVersion()) + " " +
          caffe2::to_string((unsigned long long)node);
    }
    return labelMap;
  };

  std::ofstream out("dump.dot");
  out << nom::converters::convertToDotString(g, nnprinter);
  out.close();
}

// Explore the graph in topological order until we hit stopping nodes. This is
// based on Khan's algorithm:
// https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
// Precondition: nodes in `current_frontier` must have satisfy `in_degree == 0`
void Explore(
    const std::vector<NodeRef>& current_frontier,
    VisitorContext* context) {
  std::queue<NodeRef> q;
  for (const auto n : current_frontier) {
    q.push(n);
  }

  while (!q.empty()) {
    auto node = q.front();
    q.pop();
    auto& info = context->infos.at(node);

    // Check if the node is supported, stop exploring further if not supported
    if (nn::is<NeuralNetOperator>(node)) {
      const auto* nn_op =
        nn::get<NeuralNetOperator>(node);
      const auto* op_def = dyn_cast<Caffe2Annotation>(nn_op->getAnnotation())->getOperatorDef();
      bool wanted = context->predicate(*op_def);
      wanted = context->find_supported ? wanted : (!wanted);
      if (!wanted) {
        context->frontier.emplace(node);
        continue;
      }
    }

    // Adding to current group
    info.group = context->group;
    info.needs_transform = context->find_supported;
    context->current_group.push_back(node);

    // Continue exploring its fanouts
    for (const auto& out_edge : node->getOutEdges()) {
      auto child_node = out_edge->head();
      auto& child_info = context->infos.at(child_node);
      if (--child_info.in_degree == 0) {
        q.push(child_node);
      }
    }
  }
}

// Note: subgraph always starts with ops and ends with tensors, except for the
// very first group, which can be all tensors
struct TransformSubgraph {
  explicit TransformSubgraph(
      std::vector<NodeRef>&& f,
      std::vector<NodeRef>&& n,
      int id,
      bool need)
      : input_nodes(std::move(f)),
        nodes(std::move(n)),
        group_id(id),
        needed(need) {}

  TransformSubgraph(TransformSubgraph&& rhs) noexcept
      : input_nodes(std::move(rhs.input_nodes)),
        nodes(std::move(rhs.nodes)),
        external_input_refs(std::move(rhs.external_input_refs)),
        external_output_refs(std::move(rhs.external_output_refs)),
        group_id(rhs.group_id),
        needed(rhs.needed) {}

  TransformSubgraph& operator=(TransformSubgraph&& rhs) noexcept {
    input_nodes = std::move(rhs.input_nodes);
    nodes = std::move(rhs.nodes);
    external_input_refs = std::move(external_input_refs);
    external_output_refs = std::move(external_output_refs);
    group_id = rhs.group_id;
    needed = rhs.needed;
    return *this;
  }

  void Print() const {
    LOG(INFO) << "Group :" << group_id;
    LOG(INFO) << "  Input Nodes: ";
    for (const auto i : input_nodes) {
      LOG(INFO) << "    " << ShowNode(i);
    }
    LOG(INFO) << "  Nodes: ";
    for (const auto i : nodes) {
      LOG(INFO) << "    " << ShowNode(i);
    }
  }

  std::vector<NodeRef> input_nodes;
  std::vector<NodeRef> nodes;
  std::unordered_map<std::string, NodeRef> external_input_refs;
  std::unordered_map<std::string, NodeRef> external_output_refs;
  int group_id{-1};
  bool needed{true};
};

caffe2::NetDef ConvertToC2Net(
    const TransformSubgraph& sub,
    const std::unordered_map<NodeRef, GroupAnnotation>& infos) {
  caffe2::NetDef net;
  for (auto node : sub.nodes) {
    if (nn::is<NeuralNetOperator>(node)) {
      const auto* nn_op = nn::get<NeuralNetOperator>(node);
      assert(isa<Caffe2Annotation>(nn_op->getAnnotation()) && "Cannot get caffe2 op from NNOp");
      const auto* op_def = dyn_cast<Caffe2Annotation>(nn_op->getAnnotation())->getOperatorDef();
      net.add_op()->CopyFrom(*op_def);
    }
  }
  for (const auto kv : sub.external_input_refs) {
    net.add_external_input(kv.first);
    VLOG(2) << "Adding external input: " << kv.first;
  }
  for (const auto& kv : sub.external_output_refs) {
    net.add_external_output(kv.first);
    VLOG(2) << "Adding external output: " << kv.first;
  }

  return net;
}

void DetectBoundaryReferences(
    TransformSubgraph* subgraph,
    const std::unordered_map<NodeRef, GroupAnnotation>& infos) {
  for (auto node: subgraph->nodes) {
    // inputs
    for (auto in_edge : node->getInEdges()) {
      auto parent_node = in_edge->tail();
      const auto& info = infos.at(parent_node);
      if (info.group != subgraph->group_id &&
          nn::is<NeuralNetData>(parent_node)) {
        const auto* nn_tensor = nn::get<const NeuralNetData>(parent_node);
        subgraph->external_input_refs.emplace(
            nn_tensor->getName(), parent_node);
      }
    }

    // outputs
    if (!nn::is<NeuralNetData>(node)) {
      continue;
    }
    for (auto child_node : nn::getConsumers(node)) {
      const auto& info = infos.at(child_node);
      if (info.group != subgraph->group_id) {
        const auto* nn_tensor = nn::get<const NeuralNetData>(node);
        subgraph->external_output_refs.emplace(nn_tensor->getName(), node);
        break;
      }
    }
  }
}

void ReplaceSubgraph(
    const TransformSubgraph& subgraph,
    caffe2::NetDef& net_opt,
    NNGraph* g) {
  // Delete the old subgraph starting from the input nodes until we hit boundary
  // tensors
  for (auto node : subgraph.nodes) {
    if (nn::is<NeuralNetData>(node) &&
        subgraph.external_output_refs.find(
            nn::get<const NeuralNetData>(node)->getName()) !=
            subgraph.external_output_refs.end()) {
      VLOG(2) << "Keeping " << ShowNode(node);
      continue;
    }
    VLOG(2) << "Deleting " << ShowNode(node);
    g->deleteNode(node);
  }

  // Convert new NetDef back to NNGraph
  std::unordered_map<std::string, NodeRef> tensor_map;
  for (const auto kv: subgraph.external_input_refs) {
    tensor_map.emplace(kv.first, kv.second);
  }
  for (const auto kv: subgraph.external_output_refs) {
    tensor_map.emplace(kv.first, kv.second);
  }
  for (auto& op : *net_opt.mutable_op()) {
    auto op_node = g->createNode();
    for (const auto& input : op.input()) {
      if (!tensor_map.count(input)) {
        tensor_map[input] = g->createNode(caffe2::make_unique<Tensor>(input));
      }

      auto tensor_node = tensor_map[input];
      g->createEdge(tensor_node, op_node);
    }

    for (const auto& output : op.output()) {
      if (!tensor_map.count(output)) {
        tensor_map[output] = g->createNode(caffe2::make_unique<Tensor>(output));
      }
      auto tensor_node = tensor_map[output];
      g->createEdge(op_node, tensor_node);
    }

    op_node->resetData(convertToOperatorDef(op));
    auto op_ref = dyn_cast<NeuralNetOperator>(op_node->data().get());
    CAFFE_ENFORCE(op_node->data());

    auto annotation = nom::util::make_unique<Caffe2Annotation>();
    annotation->setOperatorDef(&op);

    auto device_name = op.device_option().node_name();
    if (device_name != "") {
      annotation->setDevice(device_name);
    }

    op_ref->setAnnotation(std::move(annotation));
  }
}

void PruneUnrefereredNodes(NNGraph* g) {
  std::vector<NodeRef> to_delete;
  for (auto node : g->getMutableNodes()) {
    if (!nn::hasProducer(node) &&
        !nn::hasConsumer(node)) {
      to_delete.push_back(node);
    }
  }
  for (auto i : to_delete) {
    g->deleteNode(i);
  }
}

} // namespace

caffe2::NetDef OptimizeForBackend(
    caffe2::NetDef& net,
    std::function<bool(const caffe2::OperatorDef&)> supports,
    std::function<caffe2::NetDef(const caffe2::NetDef&)> transform_func) {
  auto nn = convertToNNModule(net);
  auto& dfg = nn.dataFlow;

  // Initialize the group info and figure out the external/input output
  VisitorContext context(supports);
  std::vector<NodeRef> external_inputs;
  std::vector<NodeRef> external_outputs;
  for (auto node : dfg.getMutableNodes()) {
    context.infos.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(node),
        std::forward_as_tuple(node->getInEdges().size(), -1));

    if (!nn::is<NeuralNetOperator>(node)) {
      if (!nn::hasProducer(node)) {
        external_inputs.push_back(node);
      }
      if (!nn::hasConsumer(node)) {
        external_outputs.push_back(node);
      }
    }
  }

  // Find unsupported and supported groups of nodes alernatively
  context.frontier.clear();
  context.current_group.clear();
  context.find_supported = false;
  std::vector<TransformSubgraph> subs;
  for (std::vector<NodeRef> frontier(
           external_inputs.begin(), external_inputs.end());
       !frontier.empty();
       context.find_supported = !context.find_supported) {
    Explore(frontier, &context);
    if (context.find_supported) {
    subs.emplace_back(
        std::move(frontier),
        std::move(context.current_group),
        context.group,
        context.find_supported);
    }

    frontier.assign(context.frontier.begin(), context.frontier.end());
    context.frontier.clear();
    context.current_group.clear();
    context.group++;
  }

  // Transform needed subgraphs one by one
  std::vector<caffe2::NetDef> opt_subnets;
  opt_subnets.reserve(subs.size());
  for (auto& g : subs) {
    // Generate boundary input/output edges
    DetectBoundaryReferences(&g, context.infos);

    caffe2::NetDef subnet = ConvertToC2Net(g, context.infos);
    // Transform the subgraph protobuf def, note that we can have less external
    // inputs/outputs but not more
    opt_subnets.emplace_back(transform_func(subnet));

    ReplaceSubgraph(g, opt_subnets.back(), &dfg);
  }

  // Prune dangling nodes, because after transformation, some weights might be
  // absorbed
  PruneUnrefereredNodes(&dfg);

  auto new_net = convertToCaffe2Proto(nn);
  for (const auto& i: net.external_input()) {
    new_net.add_external_input(i);
  }
  for (const auto& i: net.external_output()) {
    new_net.add_external_output(i);
  }
  new_net.set_name(net.name() + "_opt");
  return new_net;
}

} // namespace opt
} // namespace caffe2
