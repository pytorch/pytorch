#include "torch/csrc/jit/export.h"
#include "torch/csrc/autograd/symbolic.h"
#include "onnx/onnx.pb.h"
#include "torch/csrc/onnx/onnx.h"

#include "torch/csrc/utils/functional.h"
#include <torch/csrc/jit/assertions.h>

#include <ATen/ATen.h>
#include <ATen/optional.h>

#include <cstring>
#include <fstream>
#include <memory>
#include <vector>
#include <string>

namespace torch { namespace jit {

namespace {

namespace onnx_torch = ::torch::onnx;
namespace onnx = ::ONNX_NAMESPACE;

std::string value_name(Value* n) {
  return n->uniqueName();
}

struct ExportContext {
  size_t num_blocks = 0;
  onnx_torch::OperatorExportTypes operator_export_type;
};

void encodeGraph(onnx::GraphProto * p_g, const std::shared_ptr<Graph> & g,
                 const std::vector<at::Tensor> & initializers,
                 ExportContext *ctx, RawDataExportMap* raw_data_export_map=nullptr);

void encodeBlock(onnx::GraphProto * p_g, Block *b,
                const std::vector<at::Tensor> & initializers,
                ExportContext *ctx, RawDataExportMap* raw_data_export_map);

void encodeTensor(onnx::TensorProto * p, const at::Tensor & tensor,
                  at::optional<std::string> external_ref={},
                  RawDataExportMap* raw_data_export_map = nullptr) {
  for(auto d : tensor.sizes()) {
    p->add_dims(d);
  }
  onnx::TensorProto_DataType onnx_type;
  // Most integral types and float16 need to be serialized as int32
  at::ScalarType cast_type = tensor.type().scalarType();
  switch(tensor.type().scalarType()) {
    case at::kDouble:
      onnx_type = onnx::TensorProto_DataType_DOUBLE;
      break;
    case at::kFloat:
      onnx_type = onnx::TensorProto_DataType_FLOAT;
      break;
    case at::kHalf:
      onnx_type = onnx::TensorProto_DataType_FLOAT16;
      cast_type = at::kInt;
      break;
    case at::kByte:
      onnx_type = onnx::TensorProto_DataType_UINT8;
      cast_type = at::kInt;
      break;
    case at::kChar:
      onnx_type = onnx::TensorProto_DataType_INT8;
      cast_type = at::kInt;
      break;
    case at::kShort:
      onnx_type = onnx::TensorProto_DataType_INT16;
      cast_type = at::kInt;
      break;
    case at::kInt:
      onnx_type = onnx::TensorProto_DataType_INT32;
      break;
    case at::kLong:
      onnx_type = onnx::TensorProto_DataType_INT64;
      break;
    default:
      AT_ERROR("unexpected tensor scalar type");
      break;
  }
  p->set_data_type(onnx_type);
  // CPU's HalfTensor doesn't have contiguous(), so first calling contiguous()
  auto t = tensor.contiguous().toBackend(at::kCPU).toType(cast_type);
  // Add a buffer to the raw_data_export_map for the caller to dump into an
  // external data store. If external_ref is not specified, we instead dump
  // the contiguous data into the protobuf itself
  if (external_ref) {
    // For now, we use the name of the tensor as the external lookup name to
    // avoid ONNX protobuf changes.
    JIT_ASSERT(external_ref.value() == p->name());
    JIT_ASSERT(raw_data_export_map != nullptr);
    JIT_ASSERT(raw_data_export_map->count(external_ref.value()) == 0);
    (*raw_data_export_map)[external_ref.value()] = t;
    p->set_raw_data("__EXTERNAL");
  } else {
    JIT_ASSERT(t.is_contiguous());
    p->set_raw_data(std::string(static_cast<char*>(t.data_ptr()),  t.type().elementSizeInBytes() * t.numel()));
  }
}

void addAttribute(onnx::NodeProto * n_p, jit::Node * n, jit::Symbol name, ExportContext *ctx) {
  auto attr = n_p->add_attribute();
  JIT_ASSERT(name.is_attr());
  attr->set_name(name.toUnqualString());
  switch(n->kindOf(name)) {
    case AttributeKind::f:
      attr->set_f(n->f(name));
      attr->set_type(onnx::AttributeProto_AttributeType_FLOAT);
      break;
    case AttributeKind::fs:
      attr->set_type(onnx::AttributeProto_AttributeType_FLOATS);
      for(auto & v : n->fs(name))
        attr->add_floats(v);
      break;
    case AttributeKind::i:
      attr->set_type(onnx::AttributeProto_AttributeType_INT);
      attr->set_i(n->i(name));
      break;
    case AttributeKind::is:
      attr->set_type(onnx::AttributeProto_AttributeType_INTS);
      for(auto & v : n->is(name))
        attr->add_ints(v);
      break;
    case AttributeKind::s:
      attr->set_type(onnx::AttributeProto_AttributeType_STRING);
      attr->set_s(n->s(name));
      break;
    case AttributeKind::ss:
      attr->set_type(onnx::AttributeProto_AttributeType_STRINGS);
      for(auto & v : n->ss(name))
        attr->add_strings(v);
      break;
    case AttributeKind::t: {
      attr->set_type(onnx::AttributeProto_AttributeType_TENSOR);
      auto t = attr->mutable_t();
      encodeTensor(t, n->t(name));
    } break;
    case AttributeKind::ts:
      attr->set_type(onnx::AttributeProto_AttributeType_TENSORS);
      for(auto & v : n->ts(name)) {
        auto t = attr->add_tensors();
        encodeTensor(t, v);
      }
      break;
    case AttributeKind::g: {
      attr->set_type(onnx::AttributeProto_AttributeType_GRAPH);
      auto g = attr->mutable_g();
      encodeGraph(g, n->g(name), {}, ctx, nullptr);
    } break;
    case AttributeKind::gs:
      attr->set_type(onnx::AttributeProto_AttributeType_GRAPHS);
      for(auto & v : n->gs(name)) {
        auto g = attr->add_graphs();
        encodeGraph(g, v, {}, ctx, nullptr);
      }
      break;
  }
}

void encodeTypeProtoTensorType(onnx::TypeProto_Tensor* tensor_type, Value* n) {
  onnx::TensorShapeProto* shape = tensor_type->mutable_shape();
  if (TensorTypePtr node_type = n->type()->cast<TensorType>()) {
    const std::vector<std::int64_t>& sizes = node_type->sizes();
    for (size_t i = 0; i < sizes.size(); i++) {
      shape->add_dim();
      shape->mutable_dim(i)->set_dim_value(sizes[i]);
    }
    onnx::TensorProto_DataType onnx_type;
    switch(node_type->scalarType()) {
      case at::kDouble:
        onnx_type = onnx::TensorProto_DataType_DOUBLE;
        break;
      case at::kFloat:
        onnx_type = onnx::TensorProto_DataType_FLOAT;
        break;
      case at::kHalf:
        onnx_type = onnx::TensorProto_DataType_FLOAT16;
        break;
      case at::kByte:
        onnx_type = onnx::TensorProto_DataType_UINT8;
        break;
      case at::kChar:
        onnx_type = onnx::TensorProto_DataType_INT8;
        break;
      case at::kShort:
        onnx_type = onnx::TensorProto_DataType_INT16;
        break;
      case at::kInt:
        onnx_type = onnx::TensorProto_DataType_INT32;
        break;
      case at::kLong:
        onnx_type = onnx::TensorProto_DataType_INT64;
        break;
      default:
        AT_ERROR("unexpected tensor scalar type");
        break;
    }
    tensor_type->set_elem_type(onnx_type);
  }
}

void encodeValueInfo(onnx::ValueInfoProto* v, Value* n) {
  v->set_name(value_name(n));
  onnx::TypeProto* t = v->mutable_type();
  onnx::TypeProto_Tensor* tensor_type = t->mutable_tensor_type();
  encodeTypeProtoTensorType(tensor_type, n);
}

void encodeGraph(onnx::GraphProto * p_g, const std::shared_ptr<Graph>& g,
                 const std::vector<at::Tensor> & initializers,
                 ExportContext *ctx, RawDataExportMap* raw_data_export_map) {
  encodeBlock(p_g, g->block(), initializers, ctx, raw_data_export_map);
}

void encodeBlock(onnx::GraphProto * p_g, Block *b,
                 const std::vector<at::Tensor> & initializers,
                 ExportContext *ctx, RawDataExportMap* raw_data_export_map) {
  JIT_ASSERT(p_g != nullptr);
  std::string block_name = "torch-jit-export";
  if (ctx->num_blocks) {
    block_name += std::to_string(ctx->num_blocks);
  }
  ctx->num_blocks++;
  p_g->set_name(block_name);

  for (auto input : b->inputs()) {
    onnx::ValueInfoProto* v = p_g->add_input();
    encodeValueInfo(v, input);
  }
  for (auto output : b->outputs()) {
    onnx::ValueInfoProto* v = p_g->add_output();
    encodeValueInfo(v, output);
  }
  for (auto node : b->nodes()) {
    bool is_raw_export = ctx->operator_export_type == onnx_torch::OperatorExportTypes::RAW;
    if (node->kind() == prim::Undefined && !is_raw_export) {
      // Undefined nodes are used to implement optional inputs. One
      // way to "not provide" an optional input is to create an
      // Undefined node, and pass its output as that input.
      continue;
    }
    auto p_n = p_g->add_node();
    if (node->getSourceLocation()) {
      std::stringstream ss;
      node->getSourceLocation()->highlight(ss);
      p_n->set_doc_string(ss.str());
    }
    for(auto input : node->inputs()) {
      if (input->node()->kind() == prim::Undefined && !is_raw_export) {
        p_n->add_input("");
      } else {
        p_n->add_input(value_name(input));
      }
    }
    for(auto output : node->outputs()) {
      p_n->add_output(value_name(output));
    }
    if (is_raw_export) {
      JIT_ASSERT(!node->kind().is_onnx());
      p_n->set_domain(node->kind().domainString());
    }
    else if (ctx->operator_export_type != onnx_torch::OperatorExportTypes::ONNX_ATEN_FALLBACK) {
      JIT_ASSERT(node->kind().is_onnx());
    }
    p_n->set_op_type(node->kind().toUnqualString());
    for(auto attr_name : node->attributeNames()) {
      addAttribute(p_n, node, attr_name, ctx);
    }
    if (is_raw_export && node->blocks().size() > 0) {
      auto blocks = p_n->add_attribute();
      blocks->set_name("_blocks");
      blocks->set_type(onnx::AttributeProto_AttributeType_GRAPHS);
      for (auto block : node->blocks()) {
        auto graph = blocks->add_graphs();
        encodeBlock(graph, block, initializers, ctx, raw_data_export_map);
      }
    }
    if (node->kind() == torch::jit::onnx::Loop) {
      JIT_ASSERT(node->blocks().size() == 1);

      auto body = p_n->add_attribute();
      body->set_name("body");
      body->set_type(onnx::AttributeProto_AttributeType_GRAPH);
      auto g = body->mutable_g();
      encodeBlock(g, node->blocks()[0], {}, ctx, raw_data_export_map);
    }
    if (node->kind() == torch::jit::onnx::If) {
      JIT_ASSERT(node->blocks().size() == 2);

      auto true_branch = p_n->add_attribute();
      true_branch->set_name("then_branch");
      true_branch->set_type(onnx::AttributeProto_AttributeType_GRAPH);
      auto true_g = true_branch->mutable_g();
      encodeBlock(true_g, node->blocks()[0], {}, ctx, raw_data_export_map);

      auto false_branch = p_n->add_attribute();
      false_branch->set_name("else_branch");
      false_branch->set_type(onnx::AttributeProto_AttributeType_GRAPH);
      auto false_g = false_branch->mutable_g();
      encodeBlock(false_g, node->blocks()[1], {}, ctx, raw_data_export_map);
    }
  }
  auto num_initializers = initializers.size();
  JIT_ASSERT(b->inputs().size() >= num_initializers);
  size_t inputs_count = b->inputs().size() - num_initializers;
  for (auto & tensor : initializers) {
    // TODO: stop using positions to determine which initializers
    // match to which inputs
    std::string name = p_g->input(inputs_count++).name();
    auto p = p_g->add_initializer();
    p->set_name(name);
    if (raw_data_export_map) {
      encodeTensor(p, tensor, name, raw_data_export_map);
    } else {
      encodeTensor(p, tensor, {});
    }
  }
}

void encodeModel(onnx::ModelProto* p_m, const std::shared_ptr<Graph>& g,
                 const std::vector<at::Tensor>& initializers,
                 RawDataExportMap* raw_data_export_map = nullptr,
                 onnx_torch::OperatorExportTypes operator_export_type
                   = onnx_torch::OperatorExportTypes::ONNX) {
  onnx::GraphProto* p_g = p_m->mutable_graph();
  ExportContext ctx;
  ctx.operator_export_type = operator_export_type;
  encodeGraph(p_g, g, initializers, &ctx, raw_data_export_map);
}

namespace {
std::string getNodeStackTraceString(Node* n) {
  std::stringstream ss;
  if (n->getSourceLocation()) {
    n->getSourceLocation()->highlight(ss);
  } else {
    ss << "<unknown location>";
  }
  return ss.str();
}
} // namespace

void validateGraph(const std::shared_ptr<Graph>& graph, onnx_torch::OperatorExportTypes operator_export_type) {
  for (auto node : graph->nodes()) {
      // Macro'ed so we get a marginally better line number on failed export
#define FAIL_EXPORT(name) \
      throw std::runtime_error(std::string("ONNX export failed: ") + name + "\n\nGraph we tried to export:\n" + graph->toString());
    IR_IF(node, PythonOp)
      auto py_node = static_cast<torch::jit::PythonOp*>(value);
      FAIL_EXPORT(
          "Couldn't export Python operator " + py_node->name() +
          "\n\nDefined at:\n" + getNodeStackTraceString(node))
    IR_ELSE()
      // Special error messages for certain types of operators
      if (node->kind() == aten::expand) {
        FAIL_EXPORT(
            "Could not export a broadcasted operation; ONNX likely does not support this form of broadcasting.\n\nBroadcast occurred at:\n" +
            getNodeStackTraceString(node));
      }
      if (node->kind() == prim::PackPadded || node->kind() == prim::PadPacked) {
        FAIL_EXPORT(
            "Cannot export individual pack_padded_sequence or pad_packed_sequence; these operations must occur in pairs.\n\nUsage of this operation occurred at:\n" +
            getNodeStackTraceString(node));
      }
      bool is_aten_fallback = operator_export_type == onnx_torch::OperatorExportTypes::ONNX_ATEN_FALLBACK;
      if (!node->kind().is_onnx() && !is_aten_fallback && node->kind() != prim::Undefined) {
        FAIL_EXPORT(
            "Couldn't export operator " + node->kind().toDisplayString() + "\n\nDefined at:\n" +
            getNodeStackTraceString(node));
      }
    IR_END()
#undef FAIL_EXPORT
  }
}

// Pretty printing
namespace {
constexpr char indent_char = ' ';
constexpr size_t indent_multiplier = 2;

std::string idt(size_t indent) {
  return std::string(indent * indent_multiplier, indent_char);
}

std::string nlidt(size_t indent) {
  return std::string("\n") + idt(indent);
}

void dump(const onnx::TensorProto& tensor, std::ostream& stream) {
  stream << "TensorProto shape: [";
  for (int i = 0; i < tensor.dims_size(); ++i) {
    stream << tensor.dims(i) << (i == tensor.dims_size() - 1 ? "" : " ");
  }
  stream << "]";
}

void dump(const onnx::TensorShapeProto& shape, std::ostream& stream) {
  for (int i = 0; i < shape.dim_size(); ++i) {
    auto &dim = shape.dim(i);
    if (dim.has_dim_value()) {
      stream << dim.dim_value();
    } else {
      stream << "?";
    }
    stream << (i == shape.dim_size() - 1 ? "" : " ");
  }
}

void dump(const onnx::TypeProto_Tensor& tensor_type, std::ostream& stream) {
  stream << "Tensor dims: ";
  dump(tensor_type.shape(), stream);
}

void dump(const onnx::TypeProto& type, std::ostream& stream) {
  dump(type.tensor_type(), stream);
}

void dump(const onnx::ValueInfoProto& value_info, std::ostream& stream) {
  stream << "{name: \"" << value_info.name()
         << "\", type:";
  dump(value_info.type(), stream);
  stream << "}";
}

void dump(const onnx::GraphProto& graph, std::ostream& stream, size_t indent);

void dump(const onnx::AttributeProto& attr, std::ostream& stream, size_t indent) {
  stream << "{ name: '" << attr.name() << "', type: ";
  if (attr.has_f()) {
    stream << "float, value: " << attr.f();
  } else if (attr.has_i()) {
    stream << "int, value: " << attr.i();
  } else if (attr.has_s()) {
    stream << "string, value: '" << attr.s() << "'";
  } else if (attr.has_g()) {
    stream << "graph, value:\n";
    dump(attr.g(), stream, indent+1);
    stream << nlidt(indent);
  } else if (attr.has_t()) {
    stream << "tensor, value:";
    dump(attr.t(), stream);
  } else if (attr.floats_size()) {
    stream << "floats, values: [";
    for (int i = 0; i < attr.floats_size(); ++i)
      stream << attr.floats(i) << (i == attr.floats_size() - 1 ? "" : " ");
    stream << "]";
  } else if (attr.ints_size()) {
    stream << "ints, values: [";
    for (int i = 0; i < attr.ints_size(); ++i)
      stream << attr.ints(i) << (i == attr.ints_size() - 1 ? "" : " ");
    stream << "]";
  } else if (attr.strings_size()) {
    stream << "strings, values: [";
    for (int i = 0; i < attr.strings_size(); ++i)
      stream << "'" << attr.strings(i) << "'" << (i == attr.strings_size() - 1 ? "" : " ");
    stream << "]";
  } else if (attr.tensors_size()) {
    stream << "tensors, values: [";
    for (auto& t : attr.tensors()) {
      dump(t, stream);
    }
    stream << "]";
  } else if (attr.graphs_size()) {
    stream << "graphs, values: [";
    for (auto& g : attr.graphs()) {
      dump(g, stream, indent+1);
    }
    stream << "]";
  } else {
    stream << "UNKNOWN";
  }
  stream << "}";
}

void dump(const onnx::NodeProto& node, std::ostream& stream, size_t indent) {
  stream << "Node {type: \"" << node.op_type() << "\", inputs: [";
  for (int i = 0; i < node.input_size(); ++i) {
    stream << node.input(i) << (i == node.input_size() - 1 ? "" : ",");
  }
  stream << "], outputs: [";
  for (int i = 0; i < node.output_size(); ++i) {
    stream << node.output(i) << (i == node.output_size() - 1 ? "" : ",");
  }
  stream << "], attributes: [";
  for (int i = 0; i < node.attribute_size(); ++i) {
    dump(node.attribute(i), stream, indent+1);
    stream << (i == node.attribute_size() - 1 ? "" : ",");
  }
  stream << "]}";
}

void dump(const onnx::GraphProto& graph, std::ostream& stream, size_t indent) {
  stream << idt(indent) << "GraphProto {" << nlidt(indent+1)
         << "name: \"" << graph.name() << "\"" << nlidt(indent+1)
         << "inputs: [";
  for (int i = 0; i < graph.input_size(); ++i) {
    dump(graph.input(i), stream);
    stream << (i == graph.input_size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent+1)
         << "outputs: [";
  for (int i = 0; i < graph.output_size(); ++i) {
    dump(graph.output(i), stream);
    stream << (i == graph.output_size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent+1)
         << "initializers: [";
  for (int i = 0; i < graph.initializer_size(); ++i) {
    dump(graph.initializer(i), stream);
    stream << (i == graph.initializer_size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent+1)
         << "nodes: [" << nlidt(indent+2);
  for (int i = 0; i < graph.node_size(); ++i) {
    dump(graph.node(i), stream, indent+2);
    if (i != graph.node_size() - 1) stream << "," << nlidt(indent+2);
  }
  stream << nlidt(indent+1) << "]\n" << idt(indent) << "}\n";
}

void dump(const onnx::OperatorSetIdProto& operator_set_id, std::ostream& stream) {
  stream << "OperatorSetIdProto { domain: " << operator_set_id.domain() << "}";
}

void dump(const onnx::ModelProto& model, std::ostream& stream, size_t indent) {
  stream << idt(indent)
         << "ModelProto {" << nlidt(indent+1)
         << "producer_name: \"" << model.producer_name() << "\"" << nlidt(indent+1)
         << "domain: \"" << model.domain() << "\"" << nlidt(indent+1)
         << "doc_string: \"" << model.doc_string() << "\"";
  if (model.has_graph()) {
    stream << nlidt(indent+1) << "graph:\n";
    dump(model.graph(), stream, indent+2);
  }
  if (model.opset_import_size()) {
    stream << idt(indent+1) << "opset_import: [";
    for (auto &opset_imp : model.opset_import()) {
      dump(opset_imp, stream);
    }
    stream << "],\n";
  }
  stream << idt(indent) << "}\n";
}
} // namespace

std::string prettyPrint(const onnx::ModelProto& model) {
  std::stringstream ss;
  dump(model, ss, 0);
  return ss.str();
}

}

namespace {

RawDataExportMap ToModelProto(
    const std::shared_ptr<Graph>& graph,
    const std::vector<at::Tensor> & initializers,
    int64_t onnx_opset_version,
    bool defer_weight_export,
    onnx_torch::OperatorExportTypes operator_export_type,
    onnx::ModelProto *model_proto) {
  if (operator_export_type != onnx_torch::OperatorExportTypes::RAW) {
    validateGraph(graph, operator_export_type);
  }

  model_proto->set_producer_name("pytorch");
  model_proto->set_producer_version("0.3");
  model_proto->set_ir_version(onnx::IR_VERSION);
  auto* imp = model_proto->add_opset_import();
  // This is the version of ONNX operator set we are targeting
  imp->set_version(onnx_opset_version);

  // Map {external_data_ref -> raw data} for external serialization of weights
  RawDataExportMap raw_data_export_map;

  // Set up nanopb callbacks and compute the amount of space needed to store
  // the resulting protobuf
  if (defer_weight_export) {
    encodeModel(model_proto, graph, initializers, &raw_data_export_map, operator_export_type);
  } else {
    encodeModel(model_proto, graph, initializers, nullptr, operator_export_type);
  }

  return raw_data_export_map;
}

}  // namespace


std::string PrettyPrintExportedGraph(
                        const std::shared_ptr<Graph>& graph,
                        const std::vector<at::Tensor> & initializers,
                        int64_t onnx_opset_version,
                        bool defer_weight_export,
                        ::torch::onnx::OperatorExportTypes operator_export_type) {
  ::ONNX_NAMESPACE::ModelProto model_proto;
  RawDataExportMap raw_data_export_map;
  raw_data_export_map = ToModelProto(
    graph, initializers, onnx_opset_version, defer_weight_export, operator_export_type,
    &model_proto);
  return prettyPrint(model_proto);
}

// export_raw_ir will export IR ops without turning them into ONNX ops.
// The output will use the ONNX protobuf format, but the ops will not
// conform to the ONNX op specification. Thus, the output will not
// be interpretable by a ONNX-compatible framework. However, PyTorch or
// libtorch will be able to import the IR and play it back.
std::tuple<std::string, RawDataExportMap> ExportGraph(
                        const std::shared_ptr<Graph>& graph,
                        const std::vector<at::Tensor> & initializers,
                        int64_t onnx_opset_version,
                        bool defer_weight_export,
                        ::torch::onnx::OperatorExportTypes operator_export_type) {
  ::ONNX_NAMESPACE::ModelProto model_proto;
  RawDataExportMap raw_data_export_map;
  raw_data_export_map = ToModelProto(
    graph, initializers, onnx_opset_version, defer_weight_export, operator_export_type,
    &model_proto);
  return std::make_tuple(model_proto.SerializeAsString(), raw_data_export_map);
}

}}
