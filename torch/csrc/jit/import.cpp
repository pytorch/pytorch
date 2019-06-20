#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/type_resolver_util.h>

#include <ATen/core/functional.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/import.h>
#include <torch/csrc/jit/import_export_helpers.h>
#include <torch/csrc/jit/import_source.h>
#include <torch/csrc/jit/ir.h>
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
  std::vector<IValue> loadPickleArchive(const std::string& name);
  void importCallback(const std::string& qualifier);
  void moduleSetState(
      const std::shared_ptr<script::Module>& module,
      IValue state);

  caffe2::serialize::PyTorchStreamReader reader_;
  // this is a hack to make sure the script module created in C++ is the
  // same as created in Python
  script::ModuleLookup moduleLookup_;
  c10::optional<at::Device> device_;
  std::vector<std::string> moduleStack_;

  std::vector<at::Tensor> tensor_table_;
  std::vector<IValue> pickled_ivalues_;

  std::unordered_set<std::string> imported_libs_;

  std::shared_ptr<script::Module> main_module_;
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
  C10_LOG_API_USAGE_ONCE("torch.script.load");
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
  ::google::protobuf::util::JsonParseOptions opts;
  opts.ignore_unknown_fields = true;
  auto convert_result = ::google::protobuf::util::JsonToBinaryString(
      resolver.get(),
      url_prefix + "/" + model_def.GetDescriptor()->full_name(),
      json_string,
      &binary_string,
      opts);
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
  main_module_ = module_lookup({});

  const auto& module_def = model_def.main_module();

  // Load extra files.
  for (const auto& kv : extra_files) {
    const std::string& key = "extra/" + kv.first;
    if (reader_.hasFile(key)) {
      at::DataPtr meta_ptr;
      size_t meta_size;
      std::tie(meta_ptr, meta_size) = reader_.getRecord(key);
      extra_files[kv.first] =
          std::string(static_cast<char*>(meta_ptr.get()), meta_size);
    }
  }

  loadTensorTable(&model_def);
  if (model_def.proto_version() >= 2) {
    pickled_ivalues_ = loadPickleArchive("attributes.pkl");
  }

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

std::vector<IValue> ScriptModuleDeserializer::loadPickleArchive(const std::string& name) {
  at::DataPtr attributes_ptr;
  size_t attributes_size;
  std::tie(attributes_ptr, attributes_size) =
      reader_.getRecord(name);
  Unpickler unpickler(attributes_ptr.get(), attributes_size, &tensor_table_);
  return unpickler.parse_ivalue_list();
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
        at::empty(
            {0}, at::TensorOptions(type).device(storage_it->second.device()))
            .set_(storage_it->second, tensor_proto.offset(), dims, strides);
  }
  AT_ASSERT(result.defined());

  result = autograd::make_variable(result, tensor_proto.requires_grad());

  return result;
}

void ScriptModuleDeserializer::importCallback(const std::string& qualifier) {
  if (imported_libs_.count(qualifier)) {
    return;
  }
  imported_libs_.insert(qualifier);
  std::function<void(const std::string&)> import_callback =
      [this](const std::string& qualifier) { importCallback(qualifier); };
  const std::string path = ImportExportHelpers::qualifierToPath(qualifier);
  at::DataPtr data;
  size_t size;
  std::tie(data, size) = reader_.getRecord(path);
  auto src = std::make_shared<Source>(
      std::string(static_cast<const char*>(data.get()), size), path, 0);
  script::import_libs(
      *main_module_->class_compilation_unit(),
      qualifier,
      src,
      tensor_table_,
      import_callback);
}

void ScriptModuleDeserializer::moduleSetState(
    const std::shared_ptr<script::Module>& module,
    IValue state) {
  auto setstate = module->class_compilation_unit()->find_function("__setstate__");

  TORCH_CHECK(
      setstate != nullptr,
      "Cannot call '__setstate__' method because"
      " it does not exist");

  // TODO: once modules are first class in the interpreter and methods are not
  // lowered, change this to `module->run_method("__setstate__", {state});`
  setstate->run({module->module_object(), state});
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
  script::ScriptTypeParser typeParser;
  for (int i = 0; i < module_def.attributes_size(); ++i) {
    const torch::AttributeDef& attr_def = module_def.attributes(i);
    if (module->find_buffer(attr_def.name())) {
      // TODO: handle this above so this can be removed
      continue;
    }

    IValue ivalue;
    if (attr_def.id() >= 0) {
      // attribute has no value in the table, set it to None for now. After
      // __getstate__, check that all the attributes that are not Optional
      // can't be None
      ivalue = pickled_ivalues_.at(attr_def.id());
    }

    module->register_attribute(
        attr_def.name(), typeParser.parseType(attr_def.type()), ivalue);
  }
  if (module_def.has_torchscript_arena()) {
    at::DataPtr data;
    size_t size;
    std::tie(data, size) =
        reader_.getRecord(module_def.torchscript_arena().key());
    std::string data_str(static_cast<const char*>(data.get()), size);
    auto src = std::make_shared<Source>(
        std::string(static_cast<const char*>(data.get()), size),
        module_def.torchscript_arena().key(),
        1);

    std::function<void(const std::string&)> import_callback =
        [this](const std::string& qualifier) { importCallback(qualifier); };
    script::import_methods(
        *main_module_->class_compilation_unit(),
        module,
        src,
        tensor_table_,
        import_callback);
  }

  if (module_def.has_get_state_attribute_id()) {
    moduleSetState(
        module, pickled_ivalues_.at(module_def.get_state_attribute_id()));
  }

  for (const auto& slot : module->get_attributes()) {
    // Verify that all the non-optional attributes have been initialized
    // TODO: Issue #20497
    if (slot.type()->kind() != TypeKind::OptionalType) {
      TORCH_CHECK(
          !slot.value().isNone(),
          "The field '",
          slot.name(),
          "' was left unitialized after __setstate__, but expected a ",
          "value of type '",
          slot.type()->python_str(),
          "'");
    }
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
  auto module = std::make_shared<script::Module>("TODO");

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
