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

namespace onnx = ::ONNX_NAMESPACE;

// IR graph construction

class DecoderBase {
 protected:
  virtual std::shared_ptr<Graph> buildGraph(const onnx::GraphProto& graph_proto);

  void buildBlock(const onnx::GraphProto& graph_proto, Block* block,
                   std::unordered_map<std::string, Value*>& value_map);

  void buildBlocks(const std::vector<onnx::GraphProto>& graphs_, Node* node,
                   std::unordered_map<std::string, Value*>& value_map);

  virtual void buildValue(Value* value, const onnx::ValueInfoProto& valueinfo_proto) {};

  virtual void buildIntermediateValue(Value* value, const std::string& name) {};

  at::ScalarType onnxTypeToATenType(onnx::TensorProto_DataType tensor_proto);

  virtual at::Tensor buildTensor(const onnx::TensorProto& tensor_proto);
};

at::ScalarType DecoderBase::onnxTypeToATenType(onnx::TensorProto_DataType onnx_type) {
  switch(onnx_type) {
    case onnx::TensorProto_DataType_UINT8:
      return at::kByte;
    case onnx::TensorProto_DataType_INT8:
      return at::kChar;
    case onnx::TensorProto_DataType_INT16:
      return at::kShort;
    case onnx::TensorProto_DataType_INT32:
      return at::kInt;
    case onnx::TensorProto_DataType_INT64:
      return at::kLong;
    case onnx::TensorProto_DataType_FLOAT16:
      return at::kHalf;
    case onnx::TensorProto_DataType_FLOAT:
      return at::kFloat;
    case onnx::TensorProto_DataType_DOUBLE:
      return at::kDouble;
    default:
      throw std::runtime_error("Unsupported data type");
  }
}

at::Tensor DecoderBase::buildTensor(const onnx::TensorProto& tensor_proto) {
  at::Tensor tensor = at::CPU(onnxTypeToATenType(tensor_proto.data_type())).tensor();
  std::vector<int64_t> sizes = { tensor_proto.dims().begin(), tensor_proto.dims().end() };
  tensor.resize_(sizes);

  JIT_ASSERT(
      tensor.storage()->pImpl()->size() *
          tensor.storage()->pImpl()->elementSize() ==
      tensor_proto.raw_data().size());

  std::memcpy(tensor.data_ptr(), tensor_proto.raw_data().data(), tensor_proto.raw_data().size());
  return tensor;
}

void DecoderBase::buildBlocks(
    const std::vector<onnx::GraphProto>& graphs_, Node* node,
    std::unordered_map<std::string, Value*>& value_map) {
  for (auto g_ : graphs_) {
    auto block = node->addBlock();
    buildBlock(g_, block, value_map);
  }
}

std::shared_ptr<Graph> DecoderBase::buildGraph(const onnx::GraphProto& graph_proto) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> value_map;

  buildBlock(graph_proto, graph->block(), value_map);

  return graph;
}

void DecoderBase::buildBlock(const onnx::GraphProto& graph_proto, Block* block,
                std::unordered_map<std::string, Value*>& value_map) {

  for (auto & input : graph_proto.input()) {
    auto value = block->addInput();
    value_map[input.name()] = value;
    buildValue(value, input);
  }

  for (auto & node_ : graph_proto.node()) {
    JIT_ASSERT(node_.op_type() != "PythonOp");

    auto node = block->owningGraph()->create(Symbol::fromDomainAndUnqualString(node_.domain(), node_.op_type()),
                                             node_.output().size());

    for (auto & attr : node_.attribute()) {
      Symbol name = Symbol::attr(attr.name());

      switch(attr.type()) {
        case onnx::AttributeProto_AttributeType_UNDEFINED:
          throw std::runtime_error("UNDEFINED attribute unsupported");
          break;
        case onnx::AttributeProto_AttributeType_FLOAT:
          node->f_(name, attr.f());
          break;
        case onnx::AttributeProto_AttributeType_INT:
          node->i_(name, attr.i());
          break;
        case onnx::AttributeProto_AttributeType_STRING:
          node->s_(name, std::move(attr.s()));
          break;
        case onnx::AttributeProto_AttributeType_TENSOR:
          node->t_(name, buildTensor(attr.t()));
          break;
        case onnx::AttributeProto_AttributeType_GRAPH:
          node->g_(name, buildGraph(attr.g()));
          break;
        case onnx::AttributeProto_AttributeType_FLOATS:
          node->fs_(name, {attr.floats().begin(), attr.floats().end()});
          break;
        case onnx::AttributeProto_AttributeType_INTS:
          node->is_(name, {attr.ints().begin(), attr.ints().end()});
          break;
        case onnx::AttributeProto_AttributeType_STRINGS:
          node->ss_(name, {attr.strings().begin(), attr.strings().end()});
          break;
        case onnx::AttributeProto_AttributeType_TENSORS:
          node->ts_(name, fmap(attr.tensors(), [this](const onnx::TensorProto& t) {
                                                 return buildTensor(t);
                                               }));
          break;
        case onnx::AttributeProto_AttributeType_GRAPHS:
          if (attr.name() == "_blocks") {
            buildBlocks({attr.graphs().begin(), attr.graphs().end()}, node, value_map);
          }
          else {
            node->gs_(name, fmap(attr.graphs(), [this](const onnx::GraphProto& g_) {
                                                  return buildGraph(g_);
                                                }));
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
      buildIntermediateValue(node->outputs()[i], node_.output(i));
    }

    block->appendNode(node);
  }

  for (auto & output : graph_proto.output()) {
    Value* v = value_map.at(output.name());
    buildValue(v, output);
    block->registerOutput(v);
  }
}

class GraphDecoder : DecoderBase {
 public:
  std::shared_ptr<Graph> decode(const std::string& serialized_graph,
                                std::vector<at::Tensor>& initializers);

  void reconstructOutputTypes(Block *b);
};

// TODO: this should be removed once we'll be able to serialize value types
void GraphDecoder::reconstructOutputTypes(Block *b) {
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

std::shared_ptr<Graph> GraphDecoder::decode(
    const std::string& serialized_graph,
    std::vector<at::Tensor>& initializers) {
  auto model_proto = onnx::ModelProto();
  model_proto.ParseFromString(serialized_graph);

  auto graph_proto = model_proto.graph();
  auto graph = buildGraph(graph_proto);
  for (auto &tensor_ : graph_proto.initializer()) {
    initializers.push_back(buildTensor(tensor_));
  }
  reconstructOutputTypes(graph->block());
  return graph;
}

class ModuleDecoder : DecoderBase {
 public:
  std::shared_ptr<script::Module> decode(
      std::shared_ptr<script::Module> root_module,
      const std::string& serialized_module,
      const std::unordered_map<std::string, std::string>& storage_map);

 private:
  virtual std::shared_ptr<Graph> buildGraph(const onnx::GraphProto& graph_proto) override;

  virtual at::Tensor buildTensor(const onnx::TensorProto& tensor_proto) override;

  TypePtr buildType(const onnx::TypeProto& type_proto);

  virtual void buildValue(Value* value, const onnx::ValueInfoProto& valueinfo_proto) override;

  virtual void buildIntermediateValue(Value* value, const std::string& name) override;

  at::Tensor buildParameter(const onnx::TensorProto& tensor_proto);

  at::Tensor buildTensorCommon(const onnx::TensorProto& tensor_proto,
                               const int64_t storage_offset,
                               const std::vector<int64_t>& strides);

  std::pair<std::shared_ptr<script::Module>, std::string> parseFullName(
      std::shared_ptr<script::Module> root_module,
      const std::string fullname);

  const std::unordered_map<std::string, std::string> *storage_export_map_;
  std::unordered_map<std::string, std::shared_ptr<at::Tensor>> storage_map_;
  std::unordered_map<std::string, const onnx::TypeProto*> value_type_map_;
};

std::shared_ptr<Graph> ModuleDecoder::buildGraph(const onnx::GraphProto& graph_proto) {
  for (auto &subtype : graph_proto.value_info()) {
    value_type_map_[subtype.name()] = &subtype.type();
  }
  return DecoderBase::buildGraph(graph_proto);
}

TypePtr ModuleDecoder::buildType(const onnx::TypeProto& type_proto) {
  auto tensortype_proto = type_proto.tensor_type();
  auto shape_proto = tensortype_proto.shape();
  auto kind = type_proto.denotation();
  if (kind == "DynamicType") {
    return DynamicType::get();
  } else if (kind == "TensorType") {
    // TODO: Don't use DynamicType here
    return DynamicType::get();
  } else if (kind == "TupleType") {
    std::vector<TypePtr> elems;
    for (auto &subkind : shape_proto.dim()) {
      auto it = value_type_map_.find(subkind.dim_param());
      JIT_ASSERT(it != value_type_map_.end());
      elems.push_back(buildType(*it->second));
    }
    return TupleType::create(elems);
  } else if (kind == "ListType") {
    auto subkind = shape_proto.dim(0);
    auto it = value_type_map_.find(subkind.dim_param());
    JIT_ASSERT(it != value_type_map_.end());
    return ListType::create(buildType(*it->second));
  } else if (kind == "NumberType") {
    return NumberType::get();
  } else if (kind == "FloatType") {
    return FloatType::get();
  } else if (kind == "IntType") {
    return IntType::get();
  } else if (kind == "NoneType") {
    return NoneType::get();
  } else {
    throw std::runtime_error("unexpected string for type kind");
  }
}

void ModuleDecoder::buildValue(Value* value, const onnx::ValueInfoProto& valueinfo_proto) {
  value->setType(buildType(valueinfo_proto.type()));
}

void ModuleDecoder::buildIntermediateValue(Value* value, const std::string& name) {
  auto it = value_type_map_.find(name);
  JIT_ASSERT(it != value_type_map_.end());
  value->setType(buildType(*it->second));
}

at::Tensor ModuleDecoder::buildParameter(const onnx::TensorProto& tensor_proto) {
  std::vector<int64_t> strides;
  // We've stored three other values (is_buffer, requires_grad, storage_offset) before strides; ignore them
  std::move(tensor_proto.int64_data().begin() + 3, tensor_proto.int64_data().end(), std::back_inserter(strides));
  auto tensor = buildTensorCommon(tensor_proto, /* storage_offset = */ tensor_proto.int64_data(2), strides);
  autograd::Variable var = autograd::make_variable(tensor, /* requires_grad = */ tensor_proto.int64_data(1));
  return var;
}

at::Tensor ModuleDecoder::buildTensor(const onnx::TensorProto& tensor_proto) {
  std::vector<int64_t> strides;
  // We've stored one other value (storage_offset) before strides; ignore it
  std::move(tensor_proto.int64_data().begin() + 1, tensor_proto.int64_data().end(), std::back_inserter(strides));
  return buildTensorCommon(tensor_proto, /* storage_offset = */ tensor_proto.int64_data(0), strides);
}

at::Tensor ModuleDecoder::buildTensorCommon(
    const onnx::TensorProto& tensor_proto,
    const int64_t storage_offset,
    const std::vector<int64_t>& strides) {
  // NB: storage_offset and strides are passed in separately because
  // because they are encoded differently for parameters and tensors
  auto storage_name = tensor_proto.doc_string();
  auto type = onnxTypeToATenType(tensor_proto.data_type());
  std::vector<int64_t> dims;
  std::move(tensor_proto.dims().begin(), tensor_proto.dims().end(), std::back_inserter(dims));

  // Find or create the storage
  at::Tensor *storage_tensor;
  auto storage_it = storage_map_.find(storage_name);
  if (storage_it == storage_map_.end()) {
    auto storage = std::make_shared<at::Tensor>(at::CPU(type).tensor());
    auto string_it = storage_export_map_->find(storage_name);
    JIT_ASSERT(string_it != storage_export_map_->end());
    storage->resize_({ static_cast<int64_t>(string_it->second.size()) });
    std::memcpy(storage->storage()->pImpl()->data(), string_it->second.data(), string_it->second.size());
    storage_map_.insert(std::make_pair(storage_name, storage));
    storage_tensor = storage.get();
  } else {
    storage_tensor = storage_it->second.get();
  }

  return at::CPU(onnxTypeToATenType(tensor_proto.data_type())).tensor(
      *storage_tensor->storage().get(), storage_offset, dims, strides);
}

// Given a full name of a parameter or method,
// return the parent submodule and local name
std::pair<std::shared_ptr<script::Module>, std::string> ModuleDecoder::parseFullName(
    std::shared_ptr<script::Module> root_module,
    const std::string fullname) {
  std::vector<std::string> vec;
  std::stringstream ss(fullname);
  std::string name;
  while (std::getline(ss, name, '.')) {
    vec.push_back(name);
  }

  std::shared_ptr<script::Module> curr = root_module;
  for (size_t i = 0; i < vec.size() - 1; i++) {
    if (curr->find_module(vec[i]) == nullptr) {
      curr->register_module(vec[i], std::make_shared<script::Module>());
    }
    curr = curr->get_module(vec[i]);
  }
  return std::make_pair(curr, vec.back());
}

std::shared_ptr<script::Module> ModuleDecoder::decode(
    const std::shared_ptr<script::Module> root_module,
    const std::string &serialized_module,
    const std::unordered_map<std::string, std::string> &storage_export_map) {
  auto model_proto = onnx::ModelProto();
  model_proto.ParseFromString(serialized_module);
  auto graph_proto = model_proto.graph();

  std::unordered_map<std::string, at::Tensor*> param_map;
  storage_export_map_ = &storage_export_map;
  storage_map_.clear();

  for (auto &tensor_proto : graph_proto.initializer()) {
    std::shared_ptr<script::Module> parent_module;
    std::string name;
    std::tie(parent_module, name) = parseFullName(root_module, tensor_proto.name());

    auto param = buildParameter(tensor_proto);
    parent_module->register_parameter(name, param, /* is_buffer = */ tensor_proto.int64_data(1));
    param_map[tensor_proto.name()] = parent_module->parameter_slot(name);
  }

  for (auto &node_proto : graph_proto.node()) {
    std::shared_ptr<script::Module> parent_module;
    std::string name;
    std::tie(parent_module, name) = parseFullName(root_module, node_proto.name());

    std::vector<at::Tensor*> member_inputs;
    for (auto &param_name : node_proto.input()) {
      member_inputs.push_back(param_map[param_name]);
    }

    auto graph = buildGraph(node_proto.attribute(0).g());
    parent_module->create_method(name, graph, member_inputs);
  }

  return root_module;
}

}  // namespace

std::shared_ptr<Graph> ImportIRGraph(const std::string& serialized_graph,
                                     std::vector<at::Tensor>& initializers) {
  GraphDecoder decoder;
  return decoder.decode(serialized_graph, initializers);
}

void ImportIRModule(
    const std::shared_ptr<script::Module> module,
    const std::string& serialized_module,
    const std::unordered_map<std::string, std::string>& storage_map) {
  ModuleDecoder decoder;
  decoder.decode(module, serialized_module, storage_map);
}

}}
