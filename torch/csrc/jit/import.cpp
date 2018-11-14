#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/type_resolver_util.h>

#include "torch/csrc/jit/import.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/operator.h"

#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/proto/torch_pb.h"
#include "caffe2/serialize/inline_container.h"
#include "onnx/onnx_pb.h"

#include <ATen/ATen.h>

#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>

namespace torch { namespace jit {

namespace {

namespace onnx = ::ONNX_NAMESPACE;

// IR graph construction

class MethodDecoder {
 public:
  MethodDecoder(
      const onnx::ModelProto& model_proto,
      const std::unordered_map<std::string, at::Tensor*>& param_map,
      script::Module* parent_module,
      std::unordered_map<uint64_t, std::shared_ptr<at::Storage>>* storage_map,
      PyTorchStreamReader* stream_reader_);

 private:
  std::shared_ptr<Graph> buildGraph(const onnx::GraphProto& graph_proto);

  void buildBlock(const onnx::GraphProto& graph_proto, Block* block,
                  std::unordered_map<std::string, Value*>& value_map);

  void buildBlocks(const std::vector<onnx::GraphProto>& graphs_, Node* node,
                   std::unordered_map<std::string, Value*>& value_map);

  void buildValue(Value* value, const onnx::ValueInfoProto& valueinfo_proto);

  void buildIntermediateValue(Value* value, const std::string& name);

  at::ScalarType onnxTypeToATenType(onnx::TensorProto_DataType tensor_proto);

  at::Tensor buildTensor(const onnx::TensorProto& tensor_proto);

  TypePtr buildType(const onnx::TypeProto& type_proto);

  at::Tensor buildTensorCommon(const onnx::TensorProto& tensor_proto,
                               const uint64_t record_number,
                               const int64_t storage_offset,
                               const std::vector<int64_t>& strides);

  std::pair<std::shared_ptr<script::Module>, std::string> parseFullName(
      ModuleLookup module_lookup,
      const std::string fullname);

  PyTorchStreamReader* stream_reader_;
  std::unordered_map<uint64_t, std::shared_ptr<at::Storage>>* storage_map_;
  std::unordered_map<std::string, const onnx::TypeProto*> value_type_map_;
};

at::ScalarType MethodDecoder::onnxTypeToATenType(
    onnx::TensorProto_DataType onnx_type) {
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

void MethodDecoder::buildBlocks(
    const std::vector<onnx::GraphProto>& graphs_,
    Node* node,
    std::unordered_map<std::string, Value*>& value_map) {
  for (auto g_ : graphs_) {
    auto block = node->addBlock();
    buildBlock(g_, block, value_map);
  }
}

std::shared_ptr<Graph> MethodDecoder::buildGraph(
    const onnx::GraphProto& graph_proto) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> value_map;

  buildBlock(graph_proto, graph->block(), value_map);

  return graph;
}

void MethodDecoder::buildBlock(
    const onnx::GraphProto& graph_proto,
    Block* block,
    std::unordered_map<std::string, Value*>& value_map) {
  for (auto &subtype : graph_proto.value_info()) {
    value_type_map_[subtype.name()] = &subtype.type();
  }

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

TypePtr MethodDecoder::buildType(const onnx::TypeProto& type_proto) {
  auto tensortype_proto = type_proto.tensor_type();
  auto shape_proto = tensortype_proto.shape();
  auto kind = type_proto.denotation();
  if (kind == "DynamicType") {
    return DynamicType::get();
  } else if (kind == "TensorType") {
    auto dims = shape_proto.dim_size();
    return TensorType::create(onnxTypeToATenType(tensortype_proto.elem_type()), -1, dims);
  } else if (kind == "CompleteTensorType") {
    // first half of the dims are sizes and the second half are strides
    auto total = shape_proto.dim_size();
    std::vector<int64_t> sizes, strides;
    for (int i = 0; i < total / 2; i++) {
      sizes.push_back(shape_proto.dim(i).dim_value());
    }
    for (int i = total / 2; i < total; i++) {
      strides.push_back(shape_proto.dim(i).dim_value());
    }
    return CompleteTensorType::create(onnxTypeToATenType(tensortype_proto.elem_type()), -1, sizes, strides);
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
  } else if (kind == "BoolType") {
    return BoolType::get();
  } else if (kind == "NoneType") {
    return NoneType::get();
  } else if (kind == "GeneratorType") {
    return GeneratorType::get();
  } else if (kind == "StringType") {
    return StringType::get();
  } else if (kind.find("OptionalType") == 0) {
    onnx::TypeProto elem_proto(type_proto);
    elem_proto.set_denotation(kind.substr(strlen("OptionalType:")));
    return OptionalType::create(buildType(elem_proto));
  } else if (kind.find("TypeVar:") == 0) {
    return VarType::create(kind.substr(strlen("TypeVar:")));
  } else {
    throw std::runtime_error("unexpected string for type kind: " + kind);
  }
}

void MethodDecoder::buildValue(
    Value* value,
    const onnx::ValueInfoProto& valueinfo_proto) {
  value->setType(buildType(valueinfo_proto.type()));
}

void MethodDecoder::buildIntermediateValue(
    Value* value,
    const std::string& name) {
  auto it = value_type_map_.find(name);
  JIT_ASSERT(it != value_type_map_.end());
  value->setType(buildType(*it->second));
}

at::Tensor MethodDecoder::buildTensor(const onnx::TensorProto& tensor_proto) {
  std::vector<int64_t> strides;
  // We've stored two other values (record no., storage_offset) before strides; ignore it
  std::move(tensor_proto.int64_data().begin() + 2, tensor_proto.int64_data().end(), std::back_inserter(strides));
  return buildTensorCommon(tensor_proto,
                           /* record_number = */ tensor_proto.int64_data(0),
                           /* storage_offset = */ tensor_proto.int64_data(1),
                           strides);
}

at::Tensor MethodDecoder::buildTensorCommon(
    const onnx::TensorProto& tensor_proto,
    const uint64_t record_number,
    const int64_t storage_offset,
    const std::vector<int64_t>& strides) {
  // NB: storage_offset and strides are passed in separately because
  // because they are encoded differently for parameters and tensors
  auto type = onnxTypeToATenType(tensor_proto.data_type());
  std::vector<int64_t> dims;
  std::move(tensor_proto.dims().begin(), tensor_proto.dims().end(), std::back_inserter(dims));

  // Find or create the storage
  auto storage_it = storage_map_->find(record_number);
  if (storage_it == storage_map_->end()) {
    at::DataPtr storage_ptr;
    int64_t size;
    std::tie(storage_ptr, size) =
        stream_reader_->getRecordWithKey(record_number);
    auto storage = std::make_shared<at::Storage>(
      at::CPU(type).typeMeta(),
      std::move(storage_ptr),
      size / at::CPU(type).typeMeta().itemsize(),
      nullptr);
    storage_map_->insert(std::make_pair(record_number, storage));
    return at::CPU(type)._th_tensor(*storage, storage_offset, dims, strides);
  }

  auto storage = storage_it->second.get();
  return at::CPU(type)._th_tensor(*storage, storage_offset, dims, strides);
}

// Given a full name of a parameter or method,
// return the parent submodule and local name
std::pair<std::shared_ptr<script::Module>, std::string> MethodDecoder::
    parseFullName(ModuleLookup module_lookup, const std::string fullname) {
  AT_ASSERT(!fullname.empty());
  std::vector<std::string> vec;
  std::stringstream ss(fullname);
  std::string name;
  while (std::getline(ss, name, '.')) {
    vec.push_back(name);
  }

  std::string last = vec.back();
  vec.pop_back();
  return std::make_pair(module_lookup(vec), std::move(last));
}

MethodDecoder::MethodDecoder(
    const onnx::ModelProto& model_proto,
    const std::unordered_map<std::string, at::Tensor*>& param_map,
    script::Module* parent_module,
    std::unordered_map<uint64_t, std::shared_ptr<at::Storage>>* storage_map,
    PyTorchStreamReader* stream_reader) {
  storage_map_ = storage_map;
  stream_reader_ = stream_reader;
  const auto& graph_proto = model_proto.graph();
  for (const auto& node_proto : graph_proto.node()) {
    std::vector<at::Tensor*> member_inputs;
    const std::string& name = node_proto.name();
    for (const auto& param_name : node_proto.input()) {
      auto it = param_map.find(param_name);
      AT_ASSERTM(it != param_map.end(), "cannot find parameter ", param_name);
      member_inputs.push_back(it->second);
    }
    auto graph = buildGraph(node_proto.attribute(0).g());
    // has_domain field has a string iff the method was optimized
    parent_module->set_optimized(node_proto.has_domain());
    parent_module->create_method(name, graph, member_inputs);
    // We store the schema in the docstring so we can parse the schema and
    // assign it to the method.
    auto schema = parseSchema(node_proto.doc_string());
    parent_module->get_method(name).setSchema(std::move(schema));
  }
}

// this is a deserializer class which loads script modules from pt files. the
// content of the file is written using PyTorchStreamWriter, for details please
// check caffe2/serialize/inline_container.h. all the records except the last
// one are tensor data, and the last record is a serialized ModelProto, defined
// in caffe2/proto/torch.proto. ModelProto contains all the metadata of the
// model, and it is serialized as json.
class ScriptModuleDeserializer final {
 public:
  ScriptModuleDeserializer(const std::string& filename)
      : ifs_(filename, std::ifstream::in | std::ifstream::binary),
        reader_(&ifs_) {
    // TODO appropriate support for mmap, right now still use stream reader
  }
  ScriptModuleDeserializer(std::istream* is) : ifs_(), reader_(is) {}
  void deserialize(ModuleLookup module_lookup) {
    torch::ModelDef model_def;
    at::DataPtr data_ptr;
    size_t data_size;
    std::tie(data_ptr, data_size) = reader_.getLastRecord();
    // NB: cannot use JsonStringToMessage, since fbcode's protobuf is too old
    // be consistent with JsonStringToMessage
    std::string url_prefix = "type.googleapis.com";
    std::unique_ptr<::google::protobuf::util::TypeResolver> resolver(
        ::google::protobuf::util::NewTypeResolverForDescriptorPool(
            url_prefix, model_def.GetDescriptor()->file()->pool()));
    std::string json_string = std::string(
        static_cast<char*>(data_ptr.get()),
        static_cast<char*>(data_ptr.get()) + data_size);
    std::string binary_string;
    auto convert_result = ::google::protobuf::util::JsonToBinaryString(
        resolver.get(),
        url_prefix + "/" + model_def.GetDescriptor()->full_name(),
        json_string,
        &binary_string);
    if (!convert_result.ok()) {
      std::stringstream ss;
      ss << convert_result;
      AT_ERROR(ss.str());
    }
    AT_ASSERTM(
        model_def.ParseFromString(binary_string),
        "JSON transcoder produced invalid protobuf output.");
    moduleLookup_ = module_lookup;
    const auto& module_def = model_def.main_module();
    collectParamsInfo(module_def, module_def.name());
    // TODO: this can be simplified when C++/Python interop lands,
    // and the submodules would be created as the same in either C++ or Python
    std::shared_ptr<script::Module> module = moduleLookup_(moduleStack_);
    convertModule(module_def, module.get());
  }

 private:
  void collectParamsInfo(
      const torch::ModuleDef& module_def,
      const std::string& prefix) {
    std::shared_ptr<script::Module> module = moduleLookup_(moduleStack_);
    for (int i = 0; i < module_def.parameters_size(); ++i) {
      const torch::ParameterDef& param_def = module_def.parameters(i);
      at::Tensor tensor = createTensor(param_def.tensor());
      autograd::Variable variable =
          autograd::make_variable(tensor, param_def.require_gradient());
      module->register_parameter(
          param_def.name(), variable, param_def.is_buffer());
      parameterMap_[prefix + param_def.name()] =
          module->parameter_slot(param_def.name());
    }
    for (int i = 0; i < module_def.submodules_size(); ++i) {
      const torch::ModuleDef& sub_def = module_def.submodules(i);
      moduleStack_.push_back(sub_def.name());
      collectParamsInfo(sub_def, prefix + sub_def.name() + ".");
      moduleStack_.pop_back();
    }
  }
  void convertModule(
      const torch::ModuleDef& module_def,
      script::Module* module) {
    for (int i = 0; i < module_def.methods_size(); ++i) {
      const torch::MethodDef& method_def = module_def.methods(i);
      // TODO read unhacked torch script, right now it's serialized onnx proto
      ::ONNX_NAMESPACE::ModelProto method_proto;
      AT_ASSERTM(
          method_proto.ParseFromString(method_def.onnx_proto()),
          "cannot parse method proto (i.e., hacked onnx proto)");
      MethodDecoder decoder(
          method_proto, parameterMap_, module, &storageMap_, &reader_);
      (void)decoder;
    }
    for (int i = 0; i < module_def.submodules_size(); ++i) {
      const torch::ModuleDef& sub_def = module_def.submodules(i);
      moduleStack_.push_back(sub_def.name());
      std::shared_ptr<script::Module> sub = moduleLookup_(moduleStack_);
      convertModule(sub_def, sub.get());
      moduleStack_.pop_back();
    }
  }
  at::Tensor createTensor(const caffe2::TensorProto& tensor_proto) {
    std::vector<int64_t> dims;
    for (int i = 0; i < tensor_proto.dims_size(); ++i) {
      dims.push_back(tensor_proto.dims(i));
    }
    AT_ASSERT(
        tensor_proto.storage_type() ==
        caffe2::TensorProto_StorageType_EXTERNAL);
    const caffe2::ExternalDataProto& external_data =
        tensor_proto.external_data();
    std::vector<int64_t> strides;
    for (int i = 0; i < external_data.strides_size(); ++i) {
      strides.push_back(external_data.strides(i));
    }
    auto type = at::typeMetaToScalarType(
        caffe2::DataTypeToTypeMeta(tensor_proto.data_type()));
    uint64_t record_id = caffe2::stoull(external_data.record_id());
    AT_ASSERT(record_id != 0);
    auto storage_it = storageMap_.find(record_id);
    if (storage_it == storageMap_.end()) {
      at::DataPtr storage_ptr;
      uint64_t record_size;
      std::tie(storage_ptr, record_size) = reader_.getRecordWithKey(record_id);
      AT_ASSERT(record_size == external_data.record_size());
      auto storage = std::make_shared<at::Storage>(
          at::CPU(type).typeMeta(),
          std::move(storage_ptr),
          record_size / at::CPU(type).typeMeta().itemsize(),
          nullptr); // NB: we didn't set any allocator for the tensor
      storageMap_.insert(std::make_pair(record_id, storage));
      return at::CPU(type)._th_tensor(
          *storage, external_data.offset(), dims, strides);
    }
    return at::CPU(type)._th_tensor(
        *(storage_it->second.get()), external_data.offset(), dims, strides);
  }
  std::ifstream ifs_;
  PyTorchStreamReader reader_;
  ModuleLookup moduleLookup_;
  std::vector<std::string> moduleStack_;
  std::unordered_map<uint64_t, std::shared_ptr<at::Storage>> storageMap_;
  std::unordered_map<std::string, at::Tensor*> parameterMap_;
};

}  // namespace

void import_ir_module(
    ModuleLookup module_lookup,
    std::istream& in) {
  ScriptModuleDeserializer deserializer(&in);
  deserializer.deserialize(module_lookup);
}

void import_ir_module(
    ModuleLookup module_lookup,
    const std::string& filename) {
  ScriptModuleDeserializer deserializer(filename);
  deserializer.deserialize(module_lookup);
}

std::shared_ptr<script::Module> load(std::istream& in) {
  auto module = std::make_shared<script::Module>();

  auto module_lookup = [&](const std::vector<std::string>& qualified_name) {
    std::shared_ptr<script::Module> curr = module;
    for (const auto& name : qualified_name) {
      if (curr->find_module(name) == nullptr) {
        curr->register_module(name, std::make_shared<script::Module>());
      }
      curr = curr->get_module(name);
    }
    return curr;
  };

  ScriptModuleDeserializer deserializer(&in);
  deserializer.deserialize(module_lookup);

  return module;
}

std::shared_ptr<script::Module> load(const std::string& filename) {
  std::ifstream in(filename, std::ios_base::binary);

  AT_CHECK(! in.fail(), "load: could not open file ", filename);

  auto module = load(in);

  return module;
}

}}
