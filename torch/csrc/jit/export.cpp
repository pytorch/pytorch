#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/type_resolver_util.h>

#include "torch/csrc/jit/export.h"
#include "torch/csrc/autograd/symbolic.h"
#include "torch/csrc/onnx/onnx.h"

#include "torch/csrc/utils/functional.h"
#include <torch/csrc/jit/assertions.h>
#include "torch/csrc/jit/passes/dead_code_elimination.h"

#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/proto/torch_pb.h"
#include "caffe2/serialize/inline_container.h"
#include "onnx/onnx_pb.h"

#include <ATen/ATen.h>
#include "c10/util/Optional.h"

#include <fstream>
#include <memory>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

namespace torch { namespace jit {

namespace {
namespace onnx_torch = ::torch::onnx;
namespace onnx = ::ONNX_NAMESPACE;

class ScriptModuleSerializer;

std::string getExportableSchemaStringForMethod(const script::Method& method) {
  const auto& schema = method.getSchema();
  for (const auto& argument : schema.arguments()) {
    AT_CHECK(
        !argument.default_value(),
        "Default arguments in script graphs may currently not be exported.");
  }
  std::ostringstream stream;
  stream << schema;
  return stream.str();
}

std::string getNodeStackTraceString(const Node* n) {
  std::stringstream ss;
  if (n->getSourceLocation()) {
    n->getSourceLocation()->highlight(ss);
  } else {
    ss << "<unknown location>";
  }
  return ss.str();
}

void validateBlock(Block *b, onnx_torch::OperatorExportTypes operator_export_type) {
  for (auto node : b->nodes()) {
    for (Block *sub_block : node->blocks()) {
      validateBlock(sub_block, operator_export_type);
    }
    // Macro'ed so we get a marginally better line number on failed export
#define FAIL_EXPORT(name) \
      throw std::runtime_error(std::string("ONNX export failed: ") + name + "\n\nGraph we tried to export:\n" + b->owningGraph()->toString());
    IR_IF(node, PythonOp)
      auto py_node = static_cast<torch::jit::PythonOp*>(value);
      FAIL_EXPORT(
          "Couldn't export Python operator " + py_node->name() +
          "\n\nDefined at:\n" + getNodeStackTraceString(node))
    IR_ELSE()
      // Special error messages for certain types of operators
      if (node->kind() == aten::expand) {
        if (operator_export_type == onnx_torch::OperatorExportTypes::ONNX_ATEN_FALLBACK) {
          WithInsertPoint guard(node);
          auto* new_node = b->owningGraph()->insertNode(
            b->owningGraph()->create(Symbol(::torch::jit::onnx::ATen), node->inputs(), node->outputs().size()));
          for (size_t i = 0; i < node->outputs().size(); ++i) {
            node->output(i)->replaceAllUsesWith(new_node->output(i));
          }
          new_node->s_(Symbol::fromQualString("attr::operator"), "expand");
        } else {
          FAIL_EXPORT(
              "Could not export a broadcasted operation; ONNX likely does not support this form of broadcasting.\n\nBroadcast occurred at:\n" +
              getNodeStackTraceString(node));
        }
      }
      if (node->kind() == prim::PackPadded || node->kind() == prim::PadPacked) {
        FAIL_EXPORT(
            "Cannot export individual pack_padded_sequence or pad_packed_sequence; these operations must occur in pairs.\n\nUsage of this operation occurred at:\n" +
            getNodeStackTraceString(node));
      }
      bool is_aten_enabled = operator_export_type ==
              onnx_torch::OperatorExportTypes::ONNX_ATEN_FALLBACK ||
          operator_export_type == onnx_torch::OperatorExportTypes::ONNX_ATEN;
      if (!node->kind().is_onnx() && !is_aten_enabled &&
          node->kind() != prim::Undefined) {
        FAIL_EXPORT(
            "Couldn't export operator " + node->kind().toDisplayString() + "\n\nDefined at:\n" +
            getNodeStackTraceString(node));
      }
    IR_END()
#undef FAIL_EXPORT
  }
}

void validateGraph(const std::shared_ptr<Graph>& graph, onnx_torch::OperatorExportTypes operator_export_type) {
  validateBlock(graph->block(), operator_export_type);
  EliminateDeadCode(graph);
}

class EncoderBase {
 public:
  EncoderBase(onnx_torch::OperatorExportTypes operator_export_type, bool strip_doc);

  onnx::ModelProto get_model_proto() {
    return model_proto_;
  }

 protected:
  void EncodeGraph(onnx::GraphProto *graph_proto,
                   const std::shared_ptr<Graph> &graph,
                   const std::vector<at::Tensor> &initializers = {});

  void EncodeBlock(onnx::GraphProto *graph_proto,
                   const Block *block,
                   const std::vector<at::Tensor> &initializers = {});

  virtual void EncodeTensor(
      onnx::TensorProto* tensor_proto,
      const at::Tensor& tensor,
      const c10::optional<std::string> external_ref = {}) = 0;

  virtual void EncodeIntermediateValueInfo(onnx::GraphProto *graph_proto,
                                           const Value* n) {};

  virtual void EncodeValueInfo(onnx::GraphProto *graph_proto,
                               onnx::ValueInfoProto* v,
                               const Value* n);

  void AddAttribute(onnx::NodeProto *node_proto, const jit::Node *node, const jit::Symbol name);

  onnx::ModelProto model_proto_;
  size_t num_blocks_;
  onnx_torch::OperatorExportTypes operator_export_type_;
  bool strip_doc_;
};

onnx::TensorProto_DataType ATenTypeToOnnxType(at::ScalarType at_type) {
  switch(at_type) {
    case at::kDouble:
      return onnx::TensorProto_DataType_DOUBLE;
    case at::kFloat:
      return onnx::TensorProto_DataType_FLOAT;
    case at::kHalf:
      return onnx::TensorProto_DataType_FLOAT16;
    case at::kByte:
      return onnx::TensorProto_DataType_UINT8;
    case at::kChar:
      return onnx::TensorProto_DataType_INT8;
    case at::kShort:
      return onnx::TensorProto_DataType_INT16;
    case at::kInt:
      return onnx::TensorProto_DataType_INT32;
    case at::kLong:
      return onnx::TensorProto_DataType_INT64;
    default:
      AT_ERROR("unexpected tensor scalar type");
  }
}

EncoderBase::EncoderBase(onnx_torch::OperatorExportTypes operator_export_type, bool strip_doc)
    : num_blocks_(0),
      operator_export_type_(operator_export_type),
      strip_doc_(strip_doc) {
  model_proto_.set_producer_name("pytorch");
  model_proto_.set_ir_version(onnx::IR_VERSION);
  model_proto_.set_producer_version("0.4");
}

void EncoderBase::EncodeValueInfo(
    onnx::GraphProto *graph_proto,
    onnx::ValueInfoProto* v,
    const Value* n) {
  v->set_name(n->uniqueName());
  onnx::TypeProto* t = v->mutable_type();
  onnx::TypeProto_Tensor* tensor_type = t->mutable_tensor_type();

  onnx::TensorShapeProto* shape = tensor_type->mutable_shape();
  if (CompleteTensorTypePtr node_type = n->type()->cast<CompleteTensorType>()) {
    const std::vector<std::int64_t>& sizes = node_type->sizes();
    for (size_t i = 0; i < sizes.size(); i++) {
      shape->add_dim();
      shape->mutable_dim(i)->set_dim_value(sizes[i]);
    }
    tensor_type->set_elem_type(ATenTypeToOnnxType(node_type->scalarType()));
  } else {
    tensor_type->set_elem_type(onnx::TensorProto_DataType_UNDEFINED);
  }
}

void EncoderBase::EncodeGraph(
    onnx::GraphProto *graph_proto,
    const std::shared_ptr<Graph> &graph,
    const std::vector<at::Tensor> &initializers) {
  EncodeBlock(graph_proto, graph->block(), initializers);
}

void EncoderBase::EncodeBlock(
    onnx::GraphProto *graph_proto, const Block *block,
    const std::vector<at::Tensor> &initializers) {
  JIT_ASSERT(graph_proto != nullptr);
  std::string block_name = "torch-jit-export";
  if (num_blocks_) {
    block_name += std::to_string(num_blocks_);
  }
  num_blocks_++;
  graph_proto->set_name(block_name);

  for (auto input : block->inputs()) {
    onnx::ValueInfoProto* v = graph_proto->add_input();
    EncodeValueInfo(graph_proto, v, input);
  }
  for (auto output : block->outputs()) {
    onnx::ValueInfoProto* v = graph_proto->add_output();
    EncodeValueInfo(graph_proto, v, output);
  }
  for (auto node : block->nodes()) {
    bool is_raw_export = operator_export_type_ == onnx_torch::OperatorExportTypes::RAW;
    if (node->kind() == prim::Undefined && !is_raw_export) {
      // Undefined nodes are used to implement optional inputs. One
      // way to "not provide" an optional input is to create an
      // Undefined node, and pass its output as that input.
      continue;
    }
    auto p_n = graph_proto->add_node();
    if (node->getSourceLocation() && !strip_doc_) {
      std::stringstream ss;
      node->getSourceLocation()->highlight(ss);
      p_n->set_doc_string(ss.str());
    }
    for(auto input : node->inputs()) {
      if (input->node()->kind() == prim::Undefined && !is_raw_export) {
        p_n->add_input("");
      } else {
        p_n->add_input(input->uniqueName());
      }
    }
    for(auto output : node->outputs()) {
      p_n->add_output(output->uniqueName());
      EncodeIntermediateValueInfo(graph_proto, output);
    }
    if (is_raw_export) {
      JIT_ASSERT(!node->kind().is_onnx());
      p_n->set_domain(node->kind().domainString());
    }
    else if (operator_export_type_ == onnx_torch::OperatorExportTypes::ONNX) {
      JIT_ASSERT(node->kind().is_onnx());
    }
    p_n->set_op_type(node->kind().toUnqualString());
    for(auto attr_name : node->attributeNames()) {
      AddAttribute(p_n, node, attr_name);
    }
    if (is_raw_export && node->blocks().size() > 0) {
      auto blocks = p_n->add_attribute();
      blocks->set_name("_blocks");
      blocks->set_type(onnx::AttributeProto_AttributeType_GRAPHS);
      for (auto block : node->blocks()) {
        auto graph = blocks->add_graphs();
        EncodeBlock(graph, block, initializers);
      }
    }
    if (node->kind() == torch::jit::onnx::Loop) {
      JIT_ASSERT(node->blocks().size() == 1);

      auto body = p_n->add_attribute();
      body->set_name("body");
      body->set_type(onnx::AttributeProto_AttributeType_GRAPH);
      auto g = body->mutable_g();
      EncodeBlock(g, node->blocks()[0]);
    }
    if (node->kind() == torch::jit::onnx::If) {
      JIT_ASSERT(node->blocks().size() == 2);

      auto true_branch = p_n->add_attribute();
      true_branch->set_name("then_branch");
      true_branch->set_type(onnx::AttributeProto_AttributeType_GRAPH);
      auto true_g = true_branch->mutable_g();
      EncodeBlock(true_g, node->blocks()[0]);

      auto false_branch = p_n->add_attribute();
      false_branch->set_name("else_branch");
      false_branch->set_type(onnx::AttributeProto_AttributeType_GRAPH);
      auto false_g = false_branch->mutable_g();
      EncodeBlock(false_g, node->blocks()[1]);
    }
  }
  auto num_initializers = initializers.size();
  JIT_ASSERT(block->inputs().size() >= num_initializers);
  size_t inputs_count = block->inputs().size() - num_initializers;
  for (auto & tensor : initializers) {
    // TODO: stop using positions to determine which initializers
    // match to which inputs
    std::string name = graph_proto->input(inputs_count++).name();
    auto p = graph_proto->add_initializer();
    p->set_name(name);
    EncodeTensor(p, tensor, name);
  }
}

void EncoderBase::AddAttribute(onnx::NodeProto *node_proto, const jit::Node *node, const jit::Symbol name) {
  auto attr = node_proto->add_attribute();
  JIT_ASSERT(name.is_attr());
  attr->set_name(name.toUnqualString());
  switch(node->kindOf(name)) {
    case AttributeKind::f:
      attr->set_f(node->f(name));
      attr->set_type(onnx::AttributeProto_AttributeType_FLOAT);
      break;
    case AttributeKind::fs:
      attr->set_type(onnx::AttributeProto_AttributeType_FLOATS);
      for(auto & v : node->fs(name))
        attr->add_floats(v);
      break;
    case AttributeKind::i:
      attr->set_type(onnx::AttributeProto_AttributeType_INT);
      attr->set_i(node->i(name));
      break;
    case AttributeKind::is:
      attr->set_type(onnx::AttributeProto_AttributeType_INTS);
      for(auto & v : node->is(name))
        attr->add_ints(v);
      break;
    case AttributeKind::s:
      attr->set_type(onnx::AttributeProto_AttributeType_STRING);
      attr->set_s(node->s(name));
      break;
    case AttributeKind::ss:
      attr->set_type(onnx::AttributeProto_AttributeType_STRINGS);
      for(auto & v : node->ss(name))
        attr->add_strings(v);
      break;
    case AttributeKind::t: {
      attr->set_type(onnx::AttributeProto_AttributeType_TENSOR);
      auto t = attr->mutable_t();
      EncodeTensor(t, node->t(name));
    } break;
    case AttributeKind::ts:
      attr->set_type(onnx::AttributeProto_AttributeType_TENSORS);
      for(auto & v : node->ts(name)) {
        auto t = attr->add_tensors();
        EncodeTensor(t, v);
      }
      break;
    case AttributeKind::g: {
      attr->set_type(onnx::AttributeProto_AttributeType_GRAPH);
      auto g = attr->mutable_g();
      EncodeGraph(g, node->g(name));
    } break;
    case AttributeKind::gs:
      attr->set_type(onnx::AttributeProto_AttributeType_GRAPHS);
      for(auto & v : node->gs(name)) {
        auto g = attr->add_graphs();
        EncodeGraph(g, v);
      }
      break;
    default:
      throw std::runtime_error("unexpected attribute kind");
  }
}

class GraphEncoder: public EncoderBase {
 public:
  GraphEncoder(const std::shared_ptr<Graph> &graph,
               int64_t onnx_opset_version,
               onnx_torch::OperatorExportTypes operator_export_type,
               const std::vector<at::Tensor> &initializers,
               bool defer_weight_export,
               bool strip_doc);

  RawDataExportMap get_raw_data_export_map() {
    return raw_data_export_map_;
  }

 private:
  void EncodeTensor(
      onnx::TensorProto* tensor_proto,
      const at::Tensor& tensor,
      const c10::optional<std::string> external_ref = {}) override;

  RawDataExportMap raw_data_export_map_;
  bool defer_weight_export_;
};

GraphEncoder::GraphEncoder(
    const std::shared_ptr<Graph> &graph,
    int64_t onnx_opset_version,
    onnx_torch::OperatorExportTypes operator_export_type,
    const std::vector<at::Tensor> &initializers,
    bool defer_weight_export,
    bool strip_doc)
    : EncoderBase(operator_export_type, strip_doc),
      defer_weight_export_(defer_weight_export) {
  if (operator_export_type != onnx_torch::OperatorExportTypes::RAW) {
    validateGraph(graph, operator_export_type);
  }

  auto* imp = model_proto_.add_opset_import();
  // This is the version of ONNX operator set we are targeting
  imp->set_version(onnx_opset_version);

  EncodeGraph(model_proto_.mutable_graph(), graph, initializers);
}

void GraphEncoder::EncodeTensor(
    onnx::TensorProto* tensor_proto,
    const at::Tensor& tensor,
    const c10::optional<std::string> external_ref) {
  for(auto d : tensor.sizes()) {
    tensor_proto->add_dims(d);
  }
  tensor_proto->set_data_type(ATenTypeToOnnxType(tensor.type().scalarType()));
  // CPU's HalfTensor doesn't have contiguous(), so first calling contiguous()
  auto t = tensor.contiguous().cpu();
  // Add a buffer to the raw_data_export_map for the caller to dump into an
  // external data store. If external_ref is not specified, we instead dump
  // the contiguous data into the protobuf itself
  if (defer_weight_export_ && external_ref) {
    // For now, we use the name of the tensor as the external lookup name to
    // avoid ONNX protobuf changes.
    JIT_ASSERT(external_ref.value() == tensor_proto->name());
    JIT_ASSERT(raw_data_export_map_.count(external_ref.value()) == 0);
    raw_data_export_map_[external_ref.value()] = t;
    tensor_proto->set_raw_data("__EXTERNAL");
  } else {
    JIT_ASSERT(t.is_contiguous());
    tensor_proto->set_raw_data(std::string(static_cast<char*>(t.data_ptr()),  t.type().elementSizeInBytes() * t.numel()));
  }
}

class MethodEncoder : public EncoderBase {
 public:
  MethodEncoder(
      const script::Method& method,
      const ScriptModuleSerializer& serializer);

  std::string EncodeMethod(
      const script::Method& method,
      const std::string& prefix);

 private:
  void EncodeTensor(
      onnx::TensorProto* tensor_proto,
      const at::Tensor& tensor,
      const c10::optional<std::string> external_ref = {}) override;

  void EncodeIntermediateValueInfo(onnx::GraphProto *graph_proto,
                                           const Value* n) override;

  void EncodeValueInfo(onnx::GraphProto *graph_proto,
                               onnx::ValueInfoProto* v,
                               const Value* n) override;

  void EncodeTypeInfo(onnx::GraphProto *graph_proto,
                      onnx::ValueInfoProto* v,
                      const TypePtr& type,
                      const std::string& name);

  // serializer already serialized all the tensors, and stores
  // the tensor and parameter tables
  const ScriptModuleSerializer* serializer_;

  // Used to create sequential dummy names for node types
  size_t type_counter_ = 0;
};

// this is a serializer class which saves script modules to pt files. the
// content of the file is written using PyTorchStreamWriter, for details please
// check caffe2/serialize/inline_container.h. all the records except the last
// one are tensor data, and the last record is a serialized ModelProto, defined
// in caffe2/proto/torch.proto. ModelProto contains all the metadata of the
// model, and it is serialized as json.
class ScriptModuleSerializer final {
 public:
  ScriptModuleSerializer(const std::string& filename);

  ScriptModuleSerializer(std::ostream* ofs);

  void serialize(const script::Module& module);

  uint64_t lookupTensorId(const at::Tensor* tensor) const;

  const std::string& lookupParamName(const at::Tensor* tensor) const;

 private:
  void convertToModel(const script::Module& module, torch::ModelDef* model_def);

  // add a tensor to the tensorTable
  void addTensor(const at::Tensor* tensor);

  // recursively collect the tensors in a block and add them to the tensorTable
  void findTensorInBlock(const Block& block);

  // recursively iterate over the whole module to collect the information of
  // tensors and parameters
  void collectInfo(const script::Module& module, const std::string& prefix);

  // write the content of the tensor to the file/stream, and save the
  // offset in the storageMap_
  void convertAndWriteTensor(
      const at::Tensor& tensor,
      caffe2::TensorProto* tensor_proto);

  // dump all the tensors in the tensorTable_ to a ModelDef (metadata) and
  // the file/stream (the content), assuming all the information of the
  // tensors has been collected. the method calls convertAndWriteTensor
  // to dump the content of a tensor
  void writeTensorTable(torch::ModelDef* model_def);

  void convertModule(
      const script::Module& module,
      const std::string& name,
      torch::ModuleDef* module_def);

  void convertParameter(
      const script::NamedParameter& param,
      torch::ParameterDef* param_def);

  void convertMethod(
      const script::Method& method,
      torch::MethodDef* method_def);

  std::ofstream ofs_;
  PyTorchStreamWriter writer_;
  // storage_ptr => record_offset
  std::unordered_map<const void*, uint64_t> storageMap_;
  // tensor => param name
  std::unordered_map<const at::Tensor*, std::string> paramMap_;
  // tensor => tensor_id
  std::unordered_map<const at::Tensor*, uint64_t> tensorTable_;
  // used for generating table id for tensors
  uint64_t nextTensorId_ = 0;
};

// MethodEncoder's methods
MethodEncoder::MethodEncoder(
    const script::Method& method,
    const ScriptModuleSerializer& serializer)
    : EncoderBase(onnx_torch::OperatorExportTypes::RAW, false) {
  serializer_ = &serializer;
}

std::string MethodEncoder::EncodeMethod(
    const script::Method& method,
    const std::string& prefix) {
  onnx::ModelProto model_proto;
  model_proto.set_doc_string("THIS PROTO IS NOT STANDARD ONNX");
  auto* node_proto = model_proto.mutable_graph()->add_node();
  node_proto->set_name(prefix + method.name());

  // We store the schema string in the docstring.
  node_proto->set_doc_string(getExportableSchemaStringForMethod(method));

  // Store member_inputs of Method in input
  for (auto& member_input : method.params()) {
    const auto& param_name = serializer_->lookupParamName(member_input);
    node_proto->add_input(param_name);
  }

  auto attr_proto = node_proto->add_attribute();
  attr_proto->set_type(onnx::AttributeProto_AttributeType_GRAPH);

  for (auto node : method.graph()->nodes()) {
    if (node->kind() == prim::PythonOp) {
      auto py_node = static_cast<torch::jit::PythonOp*>(node);
      throw std::runtime_error(
          "Couldn't export Python operator " + py_node->name() +
          "\n\nDefined at:\n" + getNodeStackTraceString(node));
    }
  }
  EncodeBlock(attr_proto->mutable_g(), method.graph()->block(), {});
  std::string torch_script;
  AT_ASSERT(model_proto.SerializeToString(&torch_script));
  return torch_script;
}

void MethodEncoder::EncodeTensor(
    onnx::TensorProto* tensor_proto,
    const at::Tensor& tensor,
    const c10::optional<std::string> external_ref) {
  uint64_t tensor_id = serializer_->lookupTensorId(&tensor);
  tensor_proto->set_name(c10::to_string(tensor_id));
  // No need to store the content of the tensor to the file/stream
  // any more, since it is already saved at the beginning of the
  // serialization in writeTensorTable
}

void MethodEncoder::EncodeIntermediateValueInfo(
    onnx::GraphProto* graph_proto,
    const Value* n) {
  auto v = graph_proto->add_value_info();
  EncodeTypeInfo(graph_proto, v, n->type(), n->uniqueName());
}

c10::optional<std::string> getBaseTypeDenotation(TypeKind& kind) {
  if (kind == TypeKind::NumberType) {
    return "NumberType";
  } else if (kind == TypeKind::FloatType) {
    return "FloatType";
  } else if (kind == TypeKind::IntType) {
    return "IntType";
  } else if (kind == TypeKind::BoolType) {
    return "BoolType";
  } else if (kind == TypeKind::NoneType) {
    return "NoneType";
  } else if (kind == TypeKind::GeneratorType) {
    return "GeneratorType";
  } else if (kind == TypeKind::StringType) {
    return "StringType";
  }
  return c10::nullopt;
}

void MethodEncoder::EncodeTypeInfo(
    onnx::GraphProto* graph_proto,
    onnx::ValueInfoProto* v,
    const TypePtr& type,
    const std::string& name) {
  v->set_name(name);
  onnx::TypeProto* type_proto = v->mutable_type();
  onnx::TypeProto_Tensor* tensortype_proto = type_proto->mutable_tensor_type();
  onnx::TensorShapeProto* shape_proto = tensortype_proto->mutable_shape();

  // Use TypeProto fields to encode types.
  // denotation stores the type as a string
  auto kind = type->kind();
  if (kind == TypeKind::DynamicType) {
    type_proto->set_denotation("DynamicType");
    tensortype_proto->set_elem_type(onnx::TensorProto_DataType_UNDEFINED);
  } else if (kind == TypeKind::TensorType) {
    type_proto->set_denotation("TensorType");
    // encode the number of dimensions by pushing that number of ones into the shape proto
    auto tensor_type = type->expect<TensorType>();
    for (int i = 0; i < tensor_type->dim(); i++) {
      shape_proto->add_dim();
      shape_proto->mutable_dim(i)->set_dim_value(1);
    }
    tensortype_proto->set_elem_type(ATenTypeToOnnxType(tensor_type->scalarType()));
  } else if (kind == TypeKind::CompleteTensorType) {
    type_proto->set_denotation("CompleteTensorType");
    CompleteTensorTypePtr node_type = type->cast<CompleteTensorType>();

    // store the sizes and strides in the dims field of TensorShapeProto
    size_t i = 0;
    for (auto &size : node_type->sizes()) {
      shape_proto->add_dim();
      shape_proto->mutable_dim(i)->set_dim_value(size);
      i++;
    }
    for (auto &stride : node_type->strides()) {
      shape_proto->add_dim();
      shape_proto->mutable_dim(i)->set_dim_value(stride);
      i++;
    }
    tensortype_proto->set_elem_type(ATenTypeToOnnxType(node_type->scalarType()));
  } else if (kind == TypeKind::TupleType) {
    type_proto->set_denotation("TupleType");
    TupleTypePtr node_type = type->cast<TupleType>();
    auto elements = node_type->elements();

    // Generate a name for and encode each subtype in the value_info field of the GraphProto.
    for (size_t i = 0; i < elements.size(); i++) {
      std::string name = "#" + std::to_string(type_counter_++);
      shape_proto->add_dim();
      shape_proto->mutable_dim(i)->set_dim_param(name);
      onnx::ValueInfoProto* subtype_proto = graph_proto->add_value_info();
      EncodeTypeInfo(graph_proto, subtype_proto, elements[i], name);
    }
  } else if (kind == TypeKind::ListType) {
    type_proto->set_denotation("ListType");
    ListTypePtr node_type = type->cast<ListType>();

    // Generate a name for and encode the subtype in the value_info field of the GraphProto.
    std::string name = "#" + std::to_string(type_counter_++);
    shape_proto->add_dim();
    shape_proto->mutable_dim(0)->set_dim_param(name);
    onnx::ValueInfoProto* subtype_proto = graph_proto->add_value_info();
    EncodeTypeInfo(graph_proto, subtype_proto, node_type->getElementType(), name);
  } else if (kind == TypeKind::VarType) {
    type_proto->set_denotation("TypeVar:" + type->expect<VarType>()->name());
  } else if (kind == TypeKind::OptionalType) {
    type_proto->set_denotation("OptionalType");
    OptionalTypePtr node_type = type->cast<OptionalType>();

    // Generate a name for and encode each subtype in the value_info field of the GraphProto.
    std::string name = "#" + std::to_string(type_counter_++);
    shape_proto->add_dim();
    shape_proto->mutable_dim(0)->set_dim_param(name);
    onnx::ValueInfoProto* subtype_proto = graph_proto->add_value_info();
    EncodeTypeInfo(graph_proto, subtype_proto, node_type->getElementType(), name);
  } else {
    auto denotation = getBaseTypeDenotation(kind);
    if (!denotation) {
      throw std::runtime_error("unexpected type kind");
    }
    type_proto->set_denotation(*denotation);
  }
}

void MethodEncoder::EncodeValueInfo(
    onnx::GraphProto* graph_proto,
    onnx::ValueInfoProto* v,
    const Value* n) {
  EncodeTypeInfo(graph_proto, v, n->type(), n->uniqueName());
}

// ScriptModuleSerializer's methods
ScriptModuleSerializer::ScriptModuleSerializer(const std::string& filename)
    : ofs_(
          filename,
          std::ofstream::out | std::ofstream::trunc | std::ofstream::binary),
      writer_(&ofs_) {
  // TODO appropriate support for mmap, right now we still use stream writer
}

ScriptModuleSerializer::ScriptModuleSerializer(std::ostream* ofs)
    : ofs_(), writer_(ofs) {}

void ScriptModuleSerializer::serialize(const script::Module& module) {
  torch::ModelDef model_def;
  convertToModel(module, &model_def);
  std::string output;
  // NB: cannot use MessageToJsonString, since fbcode's protobuf is too old
  // be consistent with MessageToJsonString
  std::string url_prefix = "type.googleapis.com";
  std::unique_ptr<::google::protobuf::util::TypeResolver> resolver(
      ::google::protobuf::util::NewTypeResolverForDescriptorPool(
          url_prefix, model_def.GetDescriptor()->file()->pool()));
  ::google::protobuf::util::Status convert_result =
      ::google::protobuf::util::BinaryToJsonString(
          resolver.get(),
          url_prefix + "/" + model_def.GetDescriptor()->full_name(),
          model_def.SerializeAsString(),
          &output);
  if (!convert_result.ok()) {
    std::stringstream ss;
    ss << convert_result;
    AT_ERROR(ss.str());
  }
  auto record_id = writer_.writeRecord(output.data(), output.size());
  AT_ASSERT(record_id != 0);
  writer_.writeEndOfFile();
}

uint64_t ScriptModuleSerializer::lookupTensorId(
    const at::Tensor* tensor) const {
  auto it = tensorTable_.find(tensor);
  AT_ASSERT(it != tensorTable_.end());
  return it->second;
}

const std::string& ScriptModuleSerializer::lookupParamName(
    const at::Tensor* tensor) const {
  auto it = paramMap_.find(tensor);
  AT_ASSERT(it != paramMap_.end());
  return it->second;
}

void ScriptModuleSerializer::convertToModel(
    const script::Module& module,
    torch::ModelDef* model_def) {
  model_def->set_name("script-model");
  model_def->set_producer_name("pytorch");
  model_def->set_producer_version("1.0"); // TODO: set the producer version
                                          // using appropriate function call
  model_def->set_proto_version(torch::ProtoVersion::PROTO_VERSION_NEWEST);
  std::string main_module_name = "";
  nextTensorId_ = 0;
  collectInfo(module, main_module_name);
  writeTensorTable(model_def);
  convertModule(module, main_module_name, model_def->mutable_main_module());
}

void ScriptModuleSerializer::addTensor(const at::Tensor* tensor) {
  if (tensorTable_.find(tensor) == tensorTable_.end()) {
    tensorTable_[tensor] = nextTensorId_;
    ++nextTensorId_;
  }
}

void ScriptModuleSerializer::findTensorInBlock(const Block& block) {
  for (auto node : block.nodes()) {
    for (auto attr_name : node->attributeNames()) {
      AT_ASSERT(attr_name.is_attr());
      switch (node->kindOf(attr_name)) {
        case AttributeKind::f:
        case AttributeKind::fs:
        case AttributeKind::i:
        case AttributeKind::is:
        case AttributeKind::s:
        case AttributeKind::ss:
          break;
        case AttributeKind::t: {
          const at::Tensor* tensor = &node->t(attr_name);
          addTensor(tensor);
        } break;
        case AttributeKind::ts: {
          for (auto& v : node->ts(attr_name)) {
            const at::Tensor* tensor = &v;
            addTensor(tensor);
          }
        } break;
        case AttributeKind::g: {
          findTensorInBlock(*node->g(attr_name)->block());
        } break;
        case AttributeKind::gs: {
          for (auto& v : node->gs(attr_name)) {
            findTensorInBlock(*v->block());
          }
        } break;
        default:
          AT_ERROR("unexpected attribute kind");
      }
    }
    for (auto b : node->blocks()) {
      findTensorInBlock(*b);
    }
  }
}

void ScriptModuleSerializer::collectInfo(
    const script::Module& module,
    const std::string& prefix) {
  for (const auto& elem : module.get_parameters()) {
    const script::NamedParameter& param = elem.value();
    paramMap_[param.slot()] = prefix + param.name;
    addTensor(param.slot());
  }
  for (const auto& elem : module.get_methods()) {
    findTensorInBlock(*elem.value()->graph()->block());
  }
  for (const auto& elem : module.get_modules()) {
    collectInfo(*elem->module, prefix + elem.key() + ".");
  }
}

void ScriptModuleSerializer::convertAndWriteTensor(
    const at::Tensor& tensor,
    caffe2::TensorProto* tensor_proto) {
  auto tensor_it = tensorTable_.find(&tensor);
  AT_ASSERT(tensor_it != tensorTable_.end());
  tensor_proto->set_name(c10::to_string(tensor_it->second));
  for (auto d : tensor.sizes()) {
    tensor_proto->add_dims(d);
  }
  tensor_proto->set_data_type(caffe2::TypeMetaToDataType(
      at::scalarTypeToTypeMeta(tensor.type().scalarType())));
  tensor_proto->set_storage_type(caffe2::TensorProto_StorageType_EXTERNAL);
  caffe2::ExternalDataProto* external_data =
      tensor_proto->mutable_external_data();
  for (auto s : tensor.strides()) {
    external_data->add_strides(s);
  }
  external_data->set_offset(tensor.storage_offset());
  uint64_t record_size =
      tensor.type().elementSizeInBytes() * tensor.storage().size();
  external_data->set_record_size(record_size);
  auto* key = tensor.storage().unsafeGetStorageImpl();
  auto storage_it = storageMap_.find(key);
  if (storage_it == storageMap_.end()) {
    // TODO HIP support
    uint64_t record_id;
    if (tensor.storage().device_type() == at::DeviceType::CUDA) {
      // NB: This new tensor is created to support cuda tensors.
      // Storages can be mutated when converting tensors from cuda to cpu,
      // and we need a cpu tensor to copy data from.
      at::Tensor t = at::getType(tensor)
                         ._th_tensor(
                             tensor.storage(),
                             /* storageOffset = */ 0,
                             /* size = */
                             {static_cast<int64_t>(tensor.storage().size())},
                             /* stride = */ {1})
                         .cpu();
      AT_ASSERT(
          t.type().elementSizeInBytes() * t.storage().size() == record_size);
      record_id = writer_.writeRecord(
          t.storage().data(),
          t.type().elementSizeInBytes() * t.storage().size());
    } else {
      record_id = writer_.writeRecord(tensor.storage().data(), record_size);
    }
    external_data->set_record_id(c10::to_string(record_id));
    storageMap_[key] = record_id;
  } else {
    external_data->set_record_id(c10::to_string(storage_it->second));
  }
  // TODO handle device case, set the device_detail and load to CUDA device
}

void ScriptModuleSerializer::writeTensorTable(torch::ModelDef* model_def) {
  // NB: we don't reserve any order for tensors in the tensorTable_
  for (const auto& kv : tensorTable_) {
    auto* tensor_proto = model_def->add_tensors();
    convertAndWriteTensor(*kv.first, tensor_proto);
  }
}

void ScriptModuleSerializer::convertModule(
    const script::Module& module,
    const std::string& name,
    torch::ModuleDef* module_def) {
  module_def->set_name(name);
  module_def->set_optimize(module.is_optimized());
  for (const auto& elem : module.get_parameters()) {
    torch::ParameterDef* param_def = module_def->add_parameters();
    convertParameter(elem.value(), param_def);
  }
  for (auto& elem : module.get_methods()) {
    torch::MethodDef* method_def = module_def->add_methods();
    convertMethod(*elem.value(), method_def);
  }
  for (const auto& elem : module.get_modules()) {
    torch::ModuleDef* sub_def = module_def->add_submodules();
    convertModule(*elem->module, elem.key(), sub_def);
  }
}

void ScriptModuleSerializer::convertParameter(
    const script::NamedParameter& param,
    torch::ParameterDef* param_def) {
  param_def->set_name(param.name);
  param_def->set_is_buffer(param.is_buffer);
  param_def->set_require_gradient(param.slot()->requires_grad());
  auto it = tensorTable_.find(param.slot());
  AT_ASSERT(it != tensorTable_.end());
  param_def->set_tensor_id(c10::to_string(it->second));
}

void ScriptModuleSerializer::convertMethod(
    const script::Method& method,
    torch::MethodDef* method_def) {
  // TODO encode the real torch script instead of ModelProto
  MethodEncoder encoder(method, *this);
  // we already keep the tree structure in the top level module,
  // so pass "" as prefix
  std::string torch_script = encoder.EncodeMethod(method, "");
  method_def->set_onnx_proto(torch_script);
}

// Pretty printing
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

std::string prettyPrint(const onnx::ModelProto& model) {
  std::stringstream ss;
  dump(model, ss, 0);
  return ss.str();
}

} // namespace

std::string PrettyPrintExportedGraph(
                        const std::shared_ptr<Graph> &graph,
                        const std::vector<at::Tensor> &initializers,
                        int64_t onnx_opset_version,
                        bool defer_weight_export,
                        ::torch::onnx::OperatorExportTypes operator_export_type,
                        bool google_printer) {
  auto graph_encoder = GraphEncoder(
    graph, onnx_opset_version, operator_export_type, initializers, defer_weight_export, true);
  if (google_printer) {
    return graph_encoder.get_model_proto().DebugString();
  }
  return prettyPrint(graph_encoder.get_model_proto());
}

// export_raw_ir will export IR ops without turning them into ONNX ops.
// The output will use the ONNX protobuf format, but the ops will not
// conform to the ONNX op specification. Thus, the output will not
// be interpretable by a ONNX-compatible framework. However, PyTorch or
// libtorch will be able to import the IR and play it back.
std::tuple<std::string, RawDataExportMap> ExportGraph(
                        const std::shared_ptr<Graph> &graph,
                        const std::vector<at::Tensor> &initializers,
                        int64_t onnx_opset_version,
                        bool defer_weight_export,
                        ::torch::onnx::OperatorExportTypes operator_export_type) {
  auto graph_encoder = GraphEncoder(
    graph, onnx_opset_version, operator_export_type, initializers, defer_weight_export, false);
  return std::make_tuple(graph_encoder.get_model_proto().SerializeAsString(),
                         graph_encoder.get_raw_data_export_map());
}

void ExportModule(const script::Module& module, std::ostream& out) {
  ScriptModuleSerializer serializer(&out);
  serializer.serialize(module);
}

void ExportModule(const script::Module& module, const std::string &filename) {
  ScriptModuleSerializer serializer(filename);
  serializer.serialize(module);
}

}}
