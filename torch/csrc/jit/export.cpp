#ifndef NO_PYTHON
#include "torch/csrc/python_headers.h"
#endif

#include "torch/csrc/jit/export.h"
#include "torch/csrc/onnx/onnx.h"
#include "torch/csrc/autograd/symbolic.h"
#include "torch/csrc/Exceptions.h"

#include "torch/csrc/utils/functional.h"
#include <ATen/ATen.h>
#include <ATen/optional.h>

#include <cstring>
#include <fstream>
#include <memory>
#include <vector>
#include <string>

namespace torch { namespace jit {

namespace {

namespace onnx = ::torch::onnx;

std::string value_name(Value* n) {
  return n->uniqueName();
}

struct ExportContext {
  size_t num_blocks = 0;
  bool export_raw_ir = false;
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
  onnx::DataType onnx_type;
  // Most integral types and float16 need to be serialized as int32
  at::ScalarType cast_type = tensor.type().scalarType();
  switch(tensor.type().scalarType()) {
    case at::kDouble:
      onnx_type = onnx::kDOUBLE;
      break;
    case at::kFloat:
      onnx_type = onnx::kFLOAT;
      break;
    case at::kHalf:
      onnx_type = onnx::kFLOAT16;
      cast_type = at::kInt;
      break;
    case at::kByte:
    case at::kChar:
      onnx_type = onnx::kINT8;
      cast_type = at::kInt;
      break;
    case at::kShort:
      onnx_type = onnx::kINT16;
      cast_type = at::kInt;
      break;
    case at::kInt:
      onnx_type = onnx::kINT32;
      break;
    case at::kLong:
      onnx_type = onnx::kINT64;
      break;
    default:
      torch::barf("unexpected tensor scalar type");
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
    JIT_ASSERT(external_ref.value() == p->get_name());
    JIT_ASSERT(raw_data_export_map != nullptr);
    JIT_ASSERT(raw_data_export_map->count(external_ref.value()) == 0);
    (*raw_data_export_map)[external_ref.value()] = t;
    p->set_external_data_present();
  } else {
    p->set_raw_data(t);
  }
}

void addAttribute(onnx::NodeProto * n_p, jit::Node * n, jit::Symbol name, ExportContext *ctx) {
  auto attr = n_p->add_attribute();
  JIT_ASSERT(name.is_attr());
  attr->set_name(name.toUnqualString());
  switch(n->kindOf(name)) {
    case AttributeKind::f:
      attr->set_f(n->f(name));
      attr->set_type(onnx::aFLOAT);
      break;
    case AttributeKind::fs:
      attr->set_type(onnx::aFLOATS);
      for(auto & v : n->fs(name))
        attr->add_floats(v);
      break;
    case AttributeKind::i:
      attr->set_type(onnx::aINT);
      attr->set_i(n->i(name));
      break;
    case AttributeKind::is:
      attr->set_type(onnx::aINTS);
      for(auto & v : n->is(name))
        attr->add_ints(v);
      break;
    case AttributeKind::s:
      attr->set_type(onnx::aSTRING);
      attr->set_s(n->s(name));
      break;
    case AttributeKind::ss:
      attr->set_type(onnx::aSTRINGS);
      for(auto & v : n->ss(name))
        attr->add_strings(v);
      break;
    case AttributeKind::t: {
      attr->set_type(onnx::aTENSOR);
      auto t = attr->mutable_t();
      encodeTensor(t, n->t(name));
    } break;
    case AttributeKind::ts:
      attr->set_type(onnx::aTENSORS);
      for(auto & v : n->ts(name)) {
        auto t = attr->add_tensors();
        encodeTensor(t, v);
      }
      break;
    case AttributeKind::g: {
      attr->set_type(onnx::aGRAPH);
      auto g = attr->mutable_g();
      encodeGraph(g, n->g(name), {}, ctx, nullptr);
    } break;
    case AttributeKind::gs:
      attr->set_type(onnx::aGRAPHS);
      for(auto & v : n->gs(name)) {
        auto g = attr->add_graphs();
        encodeGraph(g, v, {}, ctx, nullptr);
      }
      break;
  }
}

void encodeTypeProtoTensorType(onnx::TypeProtoTensor* tensor_type, Value* n) {
  onnx::TensorShapeProto* shape = tensor_type->mutable_shape();
  if (TensorType* node_type = n->type()->cast<TensorType>()) {
    const std::vector<std::int64_t>& sizes = node_type->sizes();
    for (std::int64_t s : sizes) {
      shape->add_dim(s);
    }
    onnx::DataType onnx_type;
    switch(node_type->scalarType()) {
      case at::kDouble:
        onnx_type = onnx::kDOUBLE;
        break;
      case at::kFloat:
        onnx_type = onnx::kFLOAT;
        break;
      case at::kHalf:
        onnx_type = onnx::kFLOAT16;
        break;
      case at::kByte:
      case at::kChar:
        onnx_type = onnx::kINT8;
        break;
      case at::kShort:
        onnx_type = onnx::kINT16;
        break;
      case at::kInt:
        onnx_type = onnx::kINT32;
        break;
      case at::kLong:
        onnx_type = onnx::kINT64;
        break;
      default:
        torch::barf("unexpected tensor scalar type");
        break;
    }
    tensor_type->set_data_type(onnx_type);
  }
}

void encodeValueInfo(onnx::ValueInfoProto* v, Value* n) {
  v->set_name(value_name(n));
  onnx::TypeProto* t = v->mutable_type();
  onnx::TypeProtoTensor* tensor_type = t->mutable_tensor_type();
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
    if (node->kind() == prim::Undefined && !ctx->export_raw_ir) {
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
      if (input->node()->kind() == prim::Undefined && !ctx->export_raw_ir) {
        p_n->add_input("");
      } else {
        p_n->add_input(value_name(input));
      }
    }
    for(auto output : node->outputs()) {
      p_n->add_output(value_name(output));
    }
    if (ctx->export_raw_ir) {
      JIT_ASSERT(!node->kind().is_onnx());
      p_n->set_domain(node->kind().domainString());
    }
    else {
      JIT_ASSERT(node->kind().is_onnx());
    }
    p_n->set_op_type(node->kind().toUnqualString());
    for(auto attr_name : node->attributeNames()) {
      addAttribute(p_n, node, attr_name, ctx);
    }
    if (ctx->export_raw_ir && node->blocks().size() > 0) {
      auto blocks = p_n->add_attribute();
      blocks->set_name("_blocks");
      blocks->set_type(onnx::aGRAPHS);
      for (auto block : node->blocks()) {
        auto graph = blocks->add_graphs();
        encodeBlock(graph, block, initializers, ctx, raw_data_export_map);
      }
    }
    if (node->kind() == torch::jit::onnx::Loop) {
      JIT_ASSERT(node->blocks().size() == 1);

      auto body = p_n->add_attribute();
      body->set_name("body");
      body->set_type(onnx::aGRAPH);
      auto g = body->mutable_g();
      encodeBlock(g, node->blocks()[0], {}, ctx, raw_data_export_map);
    }
    if (node->kind() == torch::jit::onnx::If) {
      JIT_ASSERT(node->blocks().size() == 2);

      auto true_branch = p_n->add_attribute();
      true_branch->set_name("then_branch");
      true_branch->set_type(onnx::aGRAPH);
      auto true_g = true_branch->mutable_g();
      encodeBlock(true_g, node->blocks()[0], {}, ctx, raw_data_export_map);

      auto false_branch = p_n->add_attribute();
      false_branch->set_name("else_branch");
      false_branch->set_type(onnx::aGRAPH);
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
    std::string name = p_g->get_input_name(inputs_count++);
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
                 bool export_raw_ir = false) {
  onnx::GraphProto* p_g = p_m->mutable_graph();
  ExportContext ctx;
  ctx.export_raw_ir = export_raw_ir;
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

void validateGraph(const std::shared_ptr<Graph>& graph) {
  for (auto node : graph->nodes()) {
      // Macro'ed so we get a marginally better line number on failed export
#define FAIL_EXPORT(name) \
      throw std::runtime_error(std::string("ONNX export failed: ") + name + "\n\nGraph we tried to export:\n" + graph->toString());
    IR_IF(node, CppOp)
      auto cpp_node = static_cast<torch::jit::CppOp*>(value);
      FAIL_EXPORT(
          "Couldn't export C++ operator " + cpp_node->name() +
          "\n\nDefined at:\n" + getNodeStackTraceString(node))
      IR_ELSEIF(PythonOp)
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
      if (!node->kind().is_onnx() && node->kind() != prim::Undefined) {
        FAIL_EXPORT(
            "Couldn't export operator " + node->kind().toDisplayString() + "\n\nDefined at:\n" +
            getNodeStackTraceString(node));
      }
    IR_END()
#undef FAIL_EXPORT
  }
}

}

namespace {

RawDataExportMap ToModelProto(
    const std::shared_ptr<Graph>& graph,
    const std::vector<at::Tensor> & initializers,
    int64_t onnx_opset_version,
    bool defer_weight_export,
    bool export_raw_ir,
    onnx::ModelProto *model_proto) {
  if (!export_raw_ir) {
    validateGraph(graph);
  }

  model_proto->set_producer_name("pytorch");
  model_proto->set_producer_version("0.3");
  auto* imp = model_proto->add_opset_import();
  // This is the version of ONNX operator set we are targeting
  imp->set_version(onnx_opset_version);

  // Map {external_data_ref -> raw data} for external serialization of weights
  RawDataExportMap raw_data_export_map;

  // Set up nanopb callbacks and compute the amount of space needed to store
  // the resulting protobuf
  if (defer_weight_export) {
    encodeModel(model_proto, graph, initializers, &raw_data_export_map, export_raw_ir);
  } else {
    encodeModel(model_proto, graph, initializers, nullptr, export_raw_ir);
  }

  return raw_data_export_map;
}

}  // namespace


std::string PrettyPrintExportedGraph(
                        const std::shared_ptr<Graph>& graph,
                        const std::vector<at::Tensor> & initializers,
                        int64_t onnx_opset_version,
                        bool defer_weight_export,
                        bool export_raw_ir) {
  ::torch::onnx::ModelProto model_proto;
  RawDataExportMap raw_data_export_map;
  raw_data_export_map = ToModelProto(
    graph, initializers, onnx_opset_version, defer_weight_export, export_raw_ir, &model_proto);
  return model_proto.prettyPrint();
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
                        bool export_raw_ir) {
  ::torch::onnx::ModelProto model_proto;
  RawDataExportMap raw_data_export_map;
  raw_data_export_map = ToModelProto(
    graph, initializers, onnx_opset_version, defer_weight_export, export_raw_ir, &model_proto);

  size_t out_size;
  pb_get_encoded_size(&out_size, onnx_ModelProto_fields, &model_proto.proto);

  // Allocate storage and export the graph
  std::string out(out_size, '\0');
  pb_ostream_t ostream = pb_ostream_from_buffer(reinterpret_cast<pb_byte_t *>(&out[0]), out_size);
  pb_encode(&ostream, onnx_ModelProto_fields, &model_proto.proto);

  return std::make_tuple(out, raw_data_export_map);
}

}}
