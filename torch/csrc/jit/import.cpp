#include "torch/csrc/jit/import.h"
#include "onnx/onnx.pb.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/jit/assertions.h"

#include <ATen/ATen.h>

#include <unordered_map>
#include <vector>
#include <string>

#include <pb_decode.h>

namespace torch { namespace jit {

namespace {

// IR graph construction

at::Tensor buildTensor(const onnx_torch::TensorProto& tensor_proto) {

  at::Tensor tensor;

  switch(tensor_proto.data_type()) {
    case onnx_torch::TensorProto_DataType_UINT8:
      tensor = at::CPU(at::kByte).tensor();
      break;
    case onnx_torch::TensorProto_DataType_INT8:
      tensor = at::CPU(at::kChar).tensor();
      break;
    case onnx_torch::TensorProto_DataType_INT16:
      tensor = at::CPU(at::kShort).tensor();
      break;
    case onnx_torch::TensorProto_DataType_INT32:
      tensor = at::CPU(at::kInt).tensor();
      break;
    case onnx_torch::TensorProto_DataType_INT64:
      tensor = at::CPU(at::kLong).tensor();
      break;
    case onnx_torch::TensorProto_DataType_FLOAT16:
      tensor = at::CPU(at::kHalf).tensor();
      break;
    case onnx_torch::TensorProto_DataType_FLOAT:
      tensor = at::CPU(at::kFloat).tensor();
      break;
    case onnx_torch::TensorProto_DataType_DOUBLE:
      tensor = at::CPU(at::kDouble).tensor();
      break;
    default:
      throw std::runtime_error("Unsupported data type");
  }

  tensor.resize_({tensor_proto.dims().begin(), tensor_proto.dims().end()});

  JIT_ASSERT(
      tensor.storage()->pImpl()->get_size() *
          tensor.storage()->pImpl()->elementSize() ==
      tensor_proto.raw_data().size());

  std::memcpy(tensor.data_ptr(), tensor_proto.raw_data().data(), tensor_proto.raw_data().size());

  return tensor;
}

void buildBlock(const onnx_torch::GraphProto& graph_proto, Block* block,
                std::unordered_map<std::string, Value*>& value_map);

void buildBlocks(const std::vector<onnx_torch::GraphProto>& graphs_, Node* node,
                 std::unordered_map<std::string, Value*>& value_map) {
  for (auto g_ : graphs_) {
    auto block = node->addBlock();
    buildBlock(g_, block, value_map);
  }
}

std::shared_ptr<Graph> buildGraph(const onnx_torch::GraphProto& graph_proto) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> value_map;

  buildBlock(graph_proto, graph->block(), value_map);

  return graph;
}

void buildBlock(const onnx_torch::GraphProto& graph_proto, Block* block,
                std::unordered_map<std::string, Value*>& value_map) {

  for (auto & input : graph_proto.input()) {
    value_map[input.name()] = block->addInput();
  }

  for (auto & node_ : graph_proto.node()) {
    JIT_ASSERT(node_.op_type() != "PythonOp");

    auto node = block->owningGraph()->create(Symbol::fromDomainAndUnqualString(node_.domain(), node_.op_type()),
                                             node_.output().size());

    for (auto & attr : node_.attribute()) {
      Symbol name = Symbol::attr(attr.name());

      switch(attr.type()) {
        case onnx_torch::AttributeProto_AttributeType_UNDEFINED:
          throw std::runtime_error("UNDEFINED attribute unsupported");
          break;
        case onnx_torch::AttributeProto_AttributeType_FLOAT:
          node->f_(name, attr.f());
          break;
        case onnx_torch::AttributeProto_AttributeType_INT:
          node->i_(name, attr.i());
          break;
        case onnx_torch::AttributeProto_AttributeType_STRING:
          node->s_(name, std::move(attr.s()));
          break;
        case onnx_torch::AttributeProto_AttributeType_TENSOR:
          node->t_(name, buildTensor(attr.t()));
          break;
        case onnx_torch::AttributeProto_AttributeType_GRAPH:
          node->g_(name, buildGraph(attr.g()));
          break;
        case onnx_torch::AttributeProto_AttributeType_FLOATS:
          node->fs_(name, {attr.floats().begin(), attr.floats().end()});
          break;
        case onnx_torch::AttributeProto_AttributeType_INTS:
          node->is_(name, {attr.ints().begin(), attr.ints().end()});
          break;
        case onnx_torch::AttributeProto_AttributeType_STRINGS:
          node->ss_(name, {attr.strings().begin(), attr.strings().end()});
          break;
        case onnx_torch::AttributeProto_AttributeType_TENSORS:
          node->ts_(name, fmap(attr.tensors(), [](const onnx_torch::TensorProto& t) { return buildTensor(t); }));
          break;
        case onnx_torch::AttributeProto_AttributeType_GRAPHS:
          if (attr.name() == "_blocks") {
            buildBlocks({attr.graphs().begin(), attr.graphs().end()}, node, value_map);
          }
          else {
            node->gs_(name, fmap(attr.graphs(), [](const onnx_torch::GraphProto& g_) { return buildGraph(g_); }));
          }
          break;
      }
    }

    for (auto & input : node_.input()) {
      auto v = value_map[input];
      node->addInput(v);
    }

    for (int i=0; i<node_.output().size(); i++) {
      value_map[node_.output(i)] = node->outputs()[i];
    }

    block->appendNode(node);
  }

  for (auto & output : graph_proto.output()) {
    Value* v = value_map.at(output.name());
    block->registerOutput(v);
  }
}

std::shared_ptr<Graph> buildGraph(const onnx_torch::GraphProto& graph_proto, std::vector<at::Tensor>& initializers) {

  auto graph = buildGraph(graph_proto);

  for (auto tensor_ : graph_proto.initializer()) {
    initializers.push_back(buildTensor(tensor_));
  }

  return graph;
}

// TODO: this should be removed once we'll be able to serialize value types
void reconstructOutputTypes(Block *b) {
  for (Node * n : b->nodes()) {
    if (n->kind() == prim::Constant) {
      switch (n->kindOf(attr::value)) {
        case AttributeKind::i:
          n->output()->setType(IntType::get());
          break;
        case AttributeKind::f:
          n->output()->setType(FloatType::get());
          break;
        case AttributeKind::is:
          n->output()->setType(ListType::ofInts());
          break;
        case AttributeKind::t:
          n->output()->setType(DynamicType::get());
          break;
        default:
          throw std::runtime_error("Unsupported case in reconstructOutputTypes. File a bug report");
      }
    } else if (n->kind() == prim::ListConstruct && n->inputs().size() > 0) {
      auto input_types = fmap(n->inputs(), [](Value *v) -> TypePtr {
        return v->node()->kind() == prim::Constant ? v->type() : nullptr;
      });
      // Check that all types are equal
      if (std::equal(std::next(input_types.begin()), input_types.end(), input_types.begin())) {
        auto elem_type = input_types[0];
        if (elem_type == IntType::get()) {
          n->output()->setType(ListType::ofInts());
        }
      }
    }
    for (Block * b : n->blocks()) {
      reconstructOutputTypes(b);
    }
  }
}

} // anonymous namespace

std::shared_ptr<Graph> ImportIRGraph(const std::string& serialized_graph,
                                     std::vector<at::Tensor>& initializers) {
  auto model_proto = onnx_torch::ModelProto();
  model_proto.ParseFromString(serialized_graph);

  auto graph = buildGraph(model_proto.graph(), initializers);

  reconstructOutputTypes(graph->block());

  return graph;
}

}}
