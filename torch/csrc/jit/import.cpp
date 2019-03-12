#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/type_resolver_util.h>

#include <ATen/core/functional.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/import.h>
#include <torch/csrc/jit/import_method.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/pickler.h>
#include <torch/csrc/jit/script/script_type_parser.h>

#include "caffe2/core/common.h"
#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/proto/torch_pb.h"
#include "caffe2/serialize/file_adapter.h"
#include "caffe2/serialize/inline_container.h"
#include "caffe2/serialize/istream_adapter.h"

#include <ATen/ATen.h>

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {

using caffe2::serialize::FileAdapter;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::ReadAdapterInterface;

namespace {

// this is a deserializer class which loads script modules from pt files. the
// content of the file is written using PyTorchStreamWriter, for details please
// check caffe2/serialize/inline_container.h. all the records except the last
// one are tensor data, and the last record is a serialized ModelProto, defined
// in caffe2/proto/torch.proto. ModelProto contains all the metadata of the
// model, and it is serialized as json.
class ScriptModuleDeserializer final {
 public:
  ScriptModuleDeserializer(const std::string& filename);
  ScriptModuleDeserializer(std::istream* is);
  explicit ScriptModuleDeserializer(std::unique_ptr<ReadAdapterInterface> rai);
  void deserialize(
      script::ModuleLookup module_lookup,
      c10::optional<at::Device> device,
      script::ExtraFilesMap& extra_files);

 private:
  at::Tensor loadTensor(
      const torch::TensorDef& tensor_proto,
      std::unordered_map<std::string, at::Storage>& storageMap);

  void convertModule(const torch::ModuleDef& module_def);

  void loadTensorTable(torch::ModelDef* model_def);
  void loadAttributeTable(const torch::ModuleDef* model_def);
  IValue loadIValue(
      uint64_t byte_offset,
      const TypePtr type,
      const at::DataPtr& storage_ptr);

  caffe2::serialize::PyTorchStreamReader reader_;
  // this is a hack to make sure the script module created in C++ is the
  // same as created in Python
  script::ModuleLookup moduleLookup_;
  c10::optional<at::Device> device_;
  std::vector<std::string> moduleStack_;

  std::vector<at::Tensor> tensor_table_;
  std::unordered_map<std::string, std::pair<TypePtr, IValue>> attribute_table_;
};

ScriptModuleDeserializer::ScriptModuleDeserializer(const std::string& filename)
    : reader_(filename.c_str()) {
  // TODO appropriate support for mmap, right now still use stream reader
}

ScriptModuleDeserializer::ScriptModuleDeserializer(std::istream* is)
    : reader_(is) {}

ScriptModuleDeserializer::ScriptModuleDeserializer(
    std::unique_ptr<ReadAdapterInterface> rai)
    : reader_(std::move(rai)) {}

void ScriptModuleDeserializer::deserialize(
    script::ModuleLookup module_lookup,
    c10::optional<at::Device> device,
    script::ExtraFilesMap& extra_files) {
  torch::ModelDef model_def;
  at::DataPtr data_ptr;
  size_t data_size;
  std::tie(data_ptr, data_size) = reader_.getRecord("model.json");
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
  device_ = device;

  const auto& module_def = model_def.main_module();

  // Load extra files.
  for (const auto& kv : extra_files) {
    const std::string& key = "extra/" + kv.first;
    at::DataPtr meta_ptr;
    size_t meta_size;
    std::tie(meta_ptr, meta_size) = reader_.getRecord(key);
    extra_files[kv.first] =
        std::string(static_cast<char*>(meta_ptr.get()), meta_size);
  }

  loadTensorTable(&model_def);
  loadAttributeTable(&module_def);
  // TODO: this can be simplified when C++/Python interop lands,
  // and the submodules would be created as the same in either C++ or Python
  convertModule(module_def);
}

void ScriptModuleDeserializer::loadTensorTable(torch::ModelDef* model_def) {
  std::unordered_map<std::string, at::Storage> storageMap;
  for (const torch::TensorDef& tensor : model_def->tensors()) {
    tensor_table_.emplace_back(loadTensor(tensor, storageMap));
  }
}

struct Depickler {
  Depickler(void* data, uint64_t size)
      : bytes_(static_cast<const uint8_t*>(data)), size_(size) {}

  const std::vector<IValue>& get_ivalue_list() {
    run();
    AT_ASSERT(stack_.size() == 1);
    return stack_[0].toGenericListRef();
  }

private:
  uint8_t readByte() {
    uint8_t val = *bytes_;
    ++bytes_;
    return val;
  }

  template<typename T>
  void read(T& var) {
    const T* data = reinterpret_cast<const T*>(bytes_);
    bytes_ += sizeof(T);
    var = *data;
  }

  // template <typename T>
  // static void* read<T>() {
  //   // auto index = pushBytes(sizeof(T));
  //   // T* ptr = reinterpret_cast<T*>(&stack_[index]);
  //   // *ptr = value;
  //   // return index;
  //   void* data = reinterpret_cast<void*>(bytes_);
  //   bytes_ += sizeof(T);
  //   return *data;
  // }

  double readBinfloat() {
    AT_ASSERT(sizeof(double) == 0)
    char float_data[8];

    // Pickle floats are big endian
    for(size_t i = 0; i < 8; ++i) {
      float_data[i] = bytes_[8 - 1 - i];
    }
    bytes_ += 8;

    return *(reinterpret_cast<double*>(float_data));
  }

  uint32_t readInt32() {
    return 23;
  }

  void run() {
    while (1) {
      uint8_t b = readByte();
      OpCode op = static_cast<OpCode>(b);
      std::cout << "Doing OP: [" << static_cast<char>(op) << "]\n";
      switch (op) {
        case OpCode::PROTO: {
          uint32_t version = readByte();
          std::cout << "pickle protocol = " << version << "\n";
          break;
        }
        case OpCode::EMPTY_LIST: {
          // generic list issue: we do not know what kind of list to create
          // can possible use some fake classes like IntList to make it still
          // loadable in python
          stack_.push_back(std::vector<IValue>());
          break;
        }
        case OpCode::BINPUT: {
          size_t pos = readByte();
          if (memo_.size() <= pos)
            memo_.resize(1 + 2 * pos);
          memo_[pos] = stack_.back();
          break;
        }
        case OpCode::MARK: {
          // Mark location of the container ivalue in the stack
          marks_.push_back(stack_.size());
          break;
        }
        // case OpCode::BININT1: {
        //   stack_.push_back(uint64_t(readByte()));
        //   break;
        // }
        case OpCode::BINUNICODE: {
          uint32_t length;
          read(length);
          const char* characters = reinterpret_cast<const char*>(bytes_);
          bytes_ += length;
          stack_.push_back(std::string(characters, /*n=*/length));
          break;
        }
        case OpCode::BINFLOAT:
          stack_.push_back(readBinfloat());
          break;
        case OpCode::TUPLE: {
          size_t start = marks_.back();
          marks_.pop_back();
          IValue tup = Tuple::create(
              std::vector<IValue>(stack_.begin() + start, stack_.end()));
          stack_.resize(start);
          stack_.push_back(tup);
        } break;
        case OpCode::EMPTY_DICT:
          stack_.push_back(c10::ivalue::UnorderedMap());
          break;
        // case OpCode::SHORT_BINSTRING: {
        //   size_t size = readByte();
        //   stack_.push_back(readString(size));
        // } break;
        case OpCode::BININT:
          // stack_.push_back(readInt32());
          break;
        case OpCode::APPENDS: {
          size_t start = marks_.back();
          marks_.pop_back();
          auto gl = stack_[start - 1].toGenericList();
          gl->elements().insert(
              gl->elements().end(), stack_.begin() + start, stack_.end());
          stack_.resize(start);
        } break;
        case OpCode::SETITEMS: {
          size_t start = marks_.back();
          marks_.pop_back();
          auto dict = stack_[start - 1].toGenericDict();
          for (size_t i = start; i < stack_.size(); i += 2) {
            dict->elements()[stack_[i]] = stack_[i + 1];
          }
          stack_.resize(start);
        } break;
        case OpCode::BINGET: {
          // stack_.push_back(memo_[reader.readInt1()]);
        } break;
        case OpCode::STOP:
          return;
        default:
          std::cout << "UNKNOWN OPCODE " << "\n";
          // return;
      }
    }
  }

  std::vector<IValue> stack_;
  std::vector<IValue> memo_;
  std::vector<size_t> marks_;
  const uint8_t* bytes_;
  uint64_t size_;
}; // namespace

void ScriptModuleDeserializer::loadAttributeTable(
    const torch::ModuleDef* module_def) {
  if (module_def->attributes_size() == 0) {
    // No attributes, no attribute file to read
    return;
  }

  at::DataPtr attributes_ptr;
  size_t attributes_size;
  std::tie(attributes_ptr, attributes_size) = reader_.getRecord("attributes");
  Depickler depickler(attributes_ptr.get(), attributes_size);
  auto& attribute_list = depickler.get_ivalue_list();
  AT_ASSERT(attribute_list.size() == size_t(module_def->attributes_size()));

  for (auto ival : attribute_list) {
    std::cout << "IVALUE: " << ival << "\n";
  }
  size_t index = 0;
  for (const torch::AttributeDef& attribute_def : module_def->attributes()) {
    auto type = script::parseType(attribute_def.type());
    attribute_table_[attribute_def.name()] =
        std::make_pair(type, attribute_list[index++]);
  }
}

/// Loads an IValue from a byte array
///
/// @param byte_offset the offset into the data array (as bytes) where the
///                    IValue begins
/// @param type the jit type of the IValue to be de-serialized
/// @return storage_ptr the data itself
IValue ScriptModuleDeserializer::loadIValue(
    uint64_t byte_offset,
    const TypePtr type,
    const at::DataPtr& storage_ptr) {
  const char* bytes = static_cast<const char*>(storage_ptr.get()) + byte_offset;

  if (auto dict_type = type->cast<DictType>()) {
    const uint64_t* data = reinterpret_cast<const uint64_t*>(bytes);
    // A byte offset into bytes that says where the mappings of key pointers
    // to value pointers start
    uint64_t dict_pairs_byte_offset = data[0];

    // Number of key value pairs in the dict
    uint64_t num_dict_entries = data[1];

    c10::ivalue::UnorderedMap result;
    const uint64_t* dict_data =
        reinterpret_cast<const uint64_t*>(bytes + dict_pairs_byte_offset);

    // Load each key/value pair and store the result
    for (size_t i = 0; i < num_dict_entries * 2; i += 2) {
      auto k =
          loadIValue(dict_data[i + 0], dict_type->getKeyType(), storage_ptr);
      auto v =
          loadIValue(dict_data[i + 1], dict_type->getValueType(), storage_ptr);
      result[k] = v;
    }
    return result;
  } else if (auto string_type = type->cast<StringType>()) {
    const uint64_t* length_ptr = reinterpret_cast<const uint64_t*>(bytes);
    const char* characters = reinterpret_cast<const char*>(length_ptr + 1);
    return std::string(characters, /*n=*/length_ptr[0]);
  } else if (auto tensor_type = type->cast<TensorType>()) {
    return tensor_table_.at(byte_offset);
  } else if (auto int_type = type->cast<IntType>()) {
    return reinterpret_cast<const int64_t*>(bytes)[0];
  }
  // TODO: types other than dict and string
  AT_ERROR("Unknown type for de-serialization:", type->python_str());
}

at::Tensor ScriptModuleDeserializer::loadTensor(
    const torch::TensorDef& tensor_proto,
    std::unordered_map<std::string, at::Storage>& storageMap) {
  std::vector<int64_t> dims(
      tensor_proto.dims().begin(), tensor_proto.dims().end());
  std::vector<int64_t> strides(
      tensor_proto.strides().begin(), tensor_proto.strides().end());
  auto type = at::typeMetaToScalarType(
      caffe2::DataTypeToTypeMeta(tensor_proto.data_type()));
  const std::string& record_key = tensor_proto.data().key();
  AT_ASSERT(tensor_proto.has_device() && !tensor_proto.device().empty());
  at::Device device(tensor_proto.device());
  if (device_.has_value()) {
    // override the device, if user provides map_location
    device = device_.value();
  }

  auto storage_it = storageMap.find(record_key);
  if (storage_it == storageMap.end()) {
    at::DataPtr storage_ptr;
    uint64_t record_size;
    std::tie(storage_ptr, record_size) = reader_.getRecord(record_key);
    auto cpu_storage = at::Storage(
        at::CPU(type).typeMeta(),
        record_size / at::CPU(type).typeMeta().itemsize(),
        std::move(storage_ptr),
        /*allocator=*/nullptr,
        /*resizable=*/false); // NB: we didn't set any allocator for the tensor
    if (device.type() == at::DeviceType::CPU) {
      storage_it =
          storageMap.insert(std::make_pair(record_key, cpu_storage)).first;
    } else if (device.type() == at::DeviceType::CUDA) {
      at::Tensor cpu_tensor =
          at::empty({0}, at::CPU(type).options()).set_(cpu_storage);
      at::Storage cuda_storage =
          cpu_tensor.to(device, cpu_tensor.scalar_type()).storage();
      storage_it =
          storageMap.insert(std::make_pair(record_key, cuda_storage)).first;
    } else {
      AT_ERROR(
          "supported devices include CPU and CUDA, however got ",
          at::DeviceTypeName(device.type(), false));
    }
  }
  if (storage_it->second.device().type() != device.type() ||
      (device.has_index() &&
       storage_it->second.device().index() != device.index())) {
    std::stringstream oss;
    oss << "storage previously was specified with device "
        << storage_it->second.device() << "but now is specified with device "
        << device << std::endl;
    AT_ERROR(oss.str());
  }

  at::Tensor result;
  if (device.type() == at::DeviceType::CPU) {
    result =
        at::empty({0}, at::CPU(type).options())
            .set_(storage_it->second, tensor_proto.offset(), dims, strides);
  } else if (device.type() == at::DeviceType::CUDA) {
    result =
        at::empty({0}, at::CUDA(type).options())
            .set_(storage_it->second, tensor_proto.offset(), dims, strides);
  }
  AT_ASSERT(result.defined());

  result = autograd::make_variable(result, tensor_proto.requires_grad());

  return result;
}

void ScriptModuleDeserializer::convertModule(
    const torch::ModuleDef& module_def) {
  std::shared_ptr<script::Module> module = moduleLookup_(moduleStack_);
  module->set_optimized(module_def.optimize());
  for (int i = 0; i < module_def.submodules_size(); ++i) {
    const torch::ModuleDef& sub_def = module_def.submodules(i);
    moduleStack_.emplace_back(sub_def.name());
    convertModule(sub_def);
    moduleStack_.pop_back();
  }
  for (int i = 0; i < module_def.parameters_size(); ++i) {
    const torch::ParameterDef& param_def = module_def.parameters(i);
    at::Tensor tensor = tensor_table_.at(param_def.tensor_id());
    if (param_def.is_buffer()) {
      module->register_buffer(param_def.name(), tensor);
    } else {
      module->register_parameter(param_def.name(), tensor, /*is_buffer=*/false);
    }
  }
  for (int i = 0; i < module_def.attributes_size(); ++i) {
    const torch::AttributeDef& attr_def = module_def.attributes(i);
    if (module->find_buffer(attr_def.name())) {
      // TODO: handle this above to this can be removed
      continue;
    }
    auto attribute = attribute_table_.at(attr_def.name());
    module->register_attribute(
        attr_def.name(),
        attribute.first,
        attribute.second);
  }
  if (module_def.has_torchscript_arena()) {
    at::DataPtr data;
    size_t size;
    std::tie(data, size) =
        reader_.getRecord(module_def.torchscript_arena().key());
    std::string data_str(static_cast<const char*>(data.get()), size);
    import_methods(module, data_str, tensor_table_);
  }
}

} // namespace

void import_ir_module(
    script::ModuleLookup module_lookup,
    std::istream& in,
    c10::optional<at::Device> device,
    script::ExtraFilesMap& extra_files) {
  ScriptModuleDeserializer deserializer(&in);
  deserializer.deserialize(module_lookup, device, extra_files);
}

void import_ir_module(
    script::ModuleLookup module_lookup,
    const std::string& filename,
    c10::optional<at::Device> device,
    script::ExtraFilesMap& extra_files) {
  ScriptModuleDeserializer deserializer(filename);
  deserializer.deserialize(module_lookup, device, extra_files);
}

void import_ir_module(
    script::ModuleLookup module_lookup,
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<at::Device> device,
    script::ExtraFilesMap& extra_files) {
  ScriptModuleDeserializer deserializer(std::move(rai));
  deserializer.deserialize(module_lookup, device, extra_files);
}

std::shared_ptr<script::Module> load(
    std::istream& in,
    c10::optional<at::Device> device,
    script::ExtraFilesMap& extra_files) {
  std::unique_ptr<IStreamAdapter> rai =
      caffe2::make_unique<IStreamAdapter>(&in);
  auto module = load(std::move(rai), device, extra_files);
  return module;
}

std::shared_ptr<script::Module> load(
    const std::string& filename,
    c10::optional<at::Device> device,
    script::ExtraFilesMap& extra_files) {
  std::unique_ptr<FileAdapter> rai = caffe2::make_unique<FileAdapter>(filename);
  auto module = load(std::move(rai), device, extra_files);
  return module;
}

std::shared_ptr<script::Module> load(
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<c10::Device> device,
    script::ExtraFilesMap& extra_files) {
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

  ScriptModuleDeserializer deserializer(std::move(rai));
  deserializer.deserialize(module_lookup, device, extra_files);

  return module;
}

} // namespace jit
} // namespace torch
