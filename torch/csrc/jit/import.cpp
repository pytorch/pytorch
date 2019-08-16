#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/type_resolver_util.h>

#include <ATen/core/functional.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/import.h>
#include <torch/csrc/jit/import_export_helpers.h>
#include <torch/csrc/jit/import_source.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/pickle.h>
#include <torch/csrc/jit/script/script_type_parser.h>
#include <torch/csrc/jit/source_range_serialization.h>
#include <torch/csrc/jit/source_range_serialization_impl.h>

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
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::ReadAdapterInterface;

namespace {

struct ClassResolver : public script::Resolver {
  explicit ClassResolver(std::shared_ptr<script::CompilationUnit> cu)
      : cu_(std::move(cu)) {}
  TypePtr resolveType(const std::string& name, const SourceRange& loc)
      const override {
    return cu_->get_type(c10::QualifiedName(name));
  }

 private:
  std::shared_ptr<script::CompilationUnit> cu_;
};

// this is a deserializer class which loads script modules from pt files. the
// content of the file is written using PyTorchStreamWriter, for details please
// check caffe2/serialize/inline_container.h. all the records except the last
// one are tensor data, and the last record is a serialized ModelProto, defined
// in caffe2/proto/torch.proto. ModelProto contains all the metadata of the
// model, and it is serialized as json.
class ScriptModuleDeserializer final {
 public:
  ScriptModuleDeserializer(
      std::shared_ptr<script::CompilationUnit> cu,
      std::unique_ptr<PyTorchStreamReader> reader)
      : compilation_unit_(cu),
        reader_(std::move(reader)) {}

  script::Module deserialize(
      c10::optional<at::Device> device,
      script::ExtraFilesMap& extra_files);

 private:
  at::Tensor loadTensor(
      const torch::TensorDef& tensor_proto,
      std::unordered_map<std::string, at::Storage>& storageMap);

  void loadTensorTable(torch::ModelDef* model_def);
  void importCallback(const std::string& qualifier);
  void moduleSetState(const script::Module& module, IValue state);

  std::shared_ptr<script::CompilationUnit> compilation_unit_;

  std::unique_ptr<PyTorchStreamReader> reader_;
  c10::optional<at::Device> device_;
  std::vector<std::string> moduleStack_;

  std::vector<at::Tensor> tensor_table_;
  std::unordered_set<std::string> imported_libs_;

  IValue LEGACY_loadPickleArchive(const std::string& name);
  script::Module LEGACY_convertModule(const torch::ModuleDef& module_def);
  std::vector<IValue> LEGACY_pickled_ivalues_;
  size_t proto_version_;
};

script::Module ScriptModuleDeserializer::deserialize(
    c10::optional<at::Device> device,
    script::ExtraFilesMap& extra_files) {
  C10_LOG_API_USAGE_ONCE("torch.script.load");
  torch::ModelDef model_def;
  at::DataPtr data_ptr;
  size_t data_size;
  std::tie(data_ptr, data_size) = reader_->getRecord("model.json");
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
  device_ = device;
  proto_version_ = model_def.proto_version();

  // Load extra files.
  for (const auto& kv : extra_files) {
    const std::string& key = "extra/" + kv.first;
    if (reader_->hasFile(key)) {
      at::DataPtr meta_ptr;
      size_t meta_size;
      std::tie(meta_ptr, meta_size) = reader_->getRecord(key);
      extra_files[kv.first] =
          std::string(static_cast<char*>(meta_ptr.get()), meta_size);
    }
  }

  loadTensorTable(&model_def);

  if (proto_version_ < 6) {
    if (proto_version_ == 2) {
      const auto& list =
          LEGACY_loadPickleArchive("attributes.pkl").toGenericList();
      LEGACY_pickled_ivalues_.insert(
          LEGACY_pickled_ivalues_.end(), list.begin(), list.end());
    } else if (proto_version_ >= 3) {
      LEGACY_pickled_ivalues_ =
          LEGACY_loadPickleArchive("attributes.pkl").toTuple()->elements();
    }
    moduleStack_.push_back("__torch__");
    const auto& module_def = model_def.main_module();
    return LEGACY_convertModule(module_def);
  } else {
    at::DataPtr pickle_ptr;
    size_t pickle_size;
    std::tie(pickle_ptr, pickle_size) = reader_->getRecord("data.pkl");

    size_t bytes_read = 0;
    auto data = reinterpret_cast<const char*>(pickle_ptr.get());
    auto reader = [&](char* buffer, size_t len) {
      if (bytes_read + len > pickle_size) {
        return false;
      }
      // Copy len bytes into buffer
      const char* start = data + bytes_read;
      std::memcpy(buffer, start, len);
      bytes_read += len;
      return true;
    };

    Unpickler unpickler(
        reader, &tensor_table_, [&](const c10::QualifiedName& qn) {
          importCallback(qn.prefix());
          return c10::StrongTypePtr(
              compilation_unit_, compilation_unit_->get_class(qn));
        });
    return script::Module(unpickler.parseModule().toObject());
  }
}

IValue ScriptModuleDeserializer::LEGACY_loadPickleArchive(
    const std::string& name) {
  at::DataPtr attributes_ptr;
  size_t attributes_size;
  std::tie(attributes_ptr, attributes_size) = reader_->getRecord(name);
  auto ivalue = unpickle(
      reinterpret_cast<const char*>(attributes_ptr.get()),
      attributes_size,
      &tensor_table_,
      [&](const c10::QualifiedName& qn) {
        importCallback(qn.prefix());
        return c10::StrongTypePtr(
            compilation_unit_, compilation_unit_->get_class(qn));
      });
  return ivalue;
}

void ScriptModuleDeserializer::loadTensorTable(torch::ModelDef* model_def) {
  std::unordered_map<std::string, at::Storage> storageMap;
  for (const torch::TensorDef& tensor : model_def->tensors()) {
    tensor_table_.emplace_back(loadTensor(tensor, storageMap));
  }
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
  if (tensor_proto.is_quantized()) {
    type = toQIntType(type);
  }
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
    std::tie(storage_ptr, record_size) = reader_->getRecord(record_key);
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
    if (tensor_proto.is_quantized()) {
      result = at::_empty_affine_quantized(
          {0},
          type,
          tensor_proto.scale(),
          tensor_proto.zero_point())
          .set_(storage_it->second, tensor_proto.offset(), dims, strides);
    }
    else {
      result =
          at::empty({0}, at::CPU(type).options())
              .set_(storage_it->second, tensor_proto.offset(), dims, strides);
    }
  } else if (device.type() == at::DeviceType::CUDA) {
    result =
        at::empty(
            {0}, c10::TensorOptions(type).device(storage_it->second.device()))
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
  const std::string path =
      ImportExportHelpers::qualifierToPath(qualifier, proto_version_);
  at::DataPtr data;
  size_t size;
  std::tie(data, size) = reader_->getRecord(path);

  std::shared_ptr<ConcreteSourceRangeUnpickler> gen_ranges = nullptr;
  if (proto_version_ >= 6) {
    at::DataPtr debug_data;
    size_t debug_size;
    std::tie(debug_data, debug_size) = reader_->getRecord(path + ".debug_pkl");

    gen_ranges = std::make_shared<ConcreteSourceRangeUnpickler>(
        std::move(debug_data), debug_size);
  }

  auto src = std::make_shared<Source>(
      std::string(static_cast<const char*>(data.get()), size),
      path,
      1,
      gen_ranges);

  script::import_libs(
      compilation_unit_, qualifier, src, tensor_table_, import_callback);
}

void ScriptModuleDeserializer::moduleSetState(
    const script::Module& module,
    IValue state) {
  auto setstate = module.find_method("__setstate__");

  TORCH_CHECK(
      setstate,
      "Cannot call '__setstate__' method because"
      " it does not exist");

  // Since all Tensors are going to be None before `__setstate__` is run, we
  // can't do any optimizations on them that depend on the module type since the
  // values aren't consistent with their corresponding types.
  GraphOptimizerEnabledGuard guard(false);

  // TODO: once modules are first class in the interpreter and methods are not
  // lowered, change this to `module->run_method("__setstate__", {state});`
  if (setstate->num_inputs() == 1) {
    setstate->run({module.module_object()});
  } else if (setstate->num_inputs() == 2) {
    setstate->run({module.module_object(), state});
  } else {
    AT_ERROR("Unexpected schema on '__setstate__'");
  }
}

script::Module ScriptModuleDeserializer::LEGACY_convertModule(
    const torch::ModuleDef& module_def) {
  // HACK: The current model exporter can create module_defs with invalid Python
  // identifiers as names (they contain `.`)
  const auto atoms = c10::QualifiedName(module_def.name()).atoms();
  const size_t numPushed = atoms.size();
  for (const auto& atom : atoms) {
    moduleStack_.emplace_back(atom);
  }
  auto module =
      script::Module(c10::QualifiedName(moduleStack_), compilation_unit_);
  for (int i = 0; i < module_def.submodules_size(); ++i) {
    const torch::ModuleDef& sub_def = module_def.submodules(i);
    auto submodule = LEGACY_convertModule(sub_def);
    module.register_module(sub_def.name(), submodule);
  }
  for (int i = 0; i < module_def.parameters_size(); ++i) {
    const torch::ParameterDef& param_def = module_def.parameters(i);
    at::Tensor tensor = tensor_table_.at(param_def.tensor_id());
    if (param_def.is_buffer()) {
      module.register_buffer(param_def.name(), tensor);
    } else {
      module.register_parameter(param_def.name(), tensor, /*is_buffer=*/false);
    }
  }
  script::ScriptTypeParser typeParser(
      std::make_shared<ClassResolver>(compilation_unit_));
  for (int i = 0; i < module_def.attributes_size(); ++i) {
    const torch::AttributeDef& attr_def = module_def.attributes(i);
    if (module.find_buffer(attr_def.name())) {
      // TODO: handle this above so this can be removed
      continue;
    }

    IValue ivalue;
    if (attr_def.id() >= 0) {
      // attribute has no value in the table, set it to None for now. After
      // __getstate__, check that all the attributes that are not Optional
      // can't be None
      ivalue = LEGACY_pickled_ivalues_.at(attr_def.id());
    }

    module.register_attribute(
        attr_def.name(), typeParser.parseType(attr_def.type()), ivalue);
  }

  // If present, load in the table of source ranges from the original
  // generating code.
  std::shared_ptr<SourceRangeUnpickler> gen_ranges = nullptr;
  if (module_def.has_torchscript_debug_arena()) {
    at::DataPtr data;
    size_t size;
    std::tie(data, size) =
        reader_->getRecord(module_def.torchscript_debug_arena().key());

    gen_ranges =
        std::make_shared<ConcreteSourceRangeUnpickler>(std::move(data), size);
  }

  if (module_def.has_torchscript_arena()) {
    at::DataPtr data;
    size_t size;
    std::tie(data, size) =
        reader_->getRecord(module_def.torchscript_arena().key());
    std::string data_str(static_cast<const char*>(data.get()), size);
    auto src = std::make_shared<Source>(
        std::string(static_cast<const char*>(data.get()), size),
        module_def.torchscript_arena().key(),
        1,
        std::move(gen_ranges));

    std::function<void(const std::string&)> import_callback =
        [&, this](const std::string& qualifier) { importCallback(qualifier); };
    script::LEGACY_import_methods(module, src, tensor_table_, import_callback);
  }

  if (module_def.has_get_state_attribute_id()) {
    moduleSetState(
        module,
        LEGACY_pickled_ivalues_.at(module_def.get_state_attribute_id()));
  }

  for (const auto& slot : module.get_attributes()) {
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

  for (size_t i = 0; i < numPushed; i++) {
    moduleStack_.pop_back();
  }
  return module;
}
} // namespace

script::Module import_ir_module(
    std::shared_ptr<script::CompilationUnit> cu,
    std::istream& in,
    c10::optional<at::Device> device,
    script::ExtraFilesMap& extra_files) {
  auto reader = torch::make_unique<PyTorchStreamReader>(&in);
  ScriptModuleDeserializer deserializer(std::move(cu), std::move(reader));
  return deserializer.deserialize(device, extra_files);
}

script::Module import_ir_module(
    std::shared_ptr<script::CompilationUnit> cu,
    const std::string& filename,
    c10::optional<at::Device> device,
    script::ExtraFilesMap& extra_files) {
  auto reader = torch::make_unique<PyTorchStreamReader>(filename);
  ScriptModuleDeserializer deserializer(std::move(cu), std::move(reader));
  return deserializer.deserialize(device, extra_files);
}

script::Module import_ir_module(
    std::shared_ptr<script::CompilationUnit> cu,
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<at::Device> device,
    script::ExtraFilesMap& extra_files) {
  auto reader = torch::make_unique<PyTorchStreamReader>(std::move(rai));
  ScriptModuleDeserializer deserializer(std::move(cu), std::move(reader));
  return deserializer.deserialize(device, extra_files);
}

script::Module load(
    std::istream& in,
    c10::optional<at::Device> device,
    script::ExtraFilesMap& extra_files) {
  std::unique_ptr<IStreamAdapter> rai =
      caffe2::make_unique<IStreamAdapter>(&in);
  auto module = load(std::move(rai), device, extra_files);
  return module;
}

script::Module load(
    const std::string& filename,
    c10::optional<at::Device> device,
    script::ExtraFilesMap& extra_files) {
  std::unique_ptr<FileAdapter> rai = caffe2::make_unique<FileAdapter>(filename);
  auto module = load(std::move(rai), device, extra_files);
  return module;
}

script::Module load(
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<c10::Device> device,
    script::ExtraFilesMap& extra_files) {
  auto reader = torch::make_unique<PyTorchStreamReader>(std::move(rai));
  auto cu = std::make_shared<script::CompilationUnit>();
  ScriptModuleDeserializer deserializer(std::move(cu), std::move(reader));
  return deserializer.deserialize(device, extra_files);
}

} // namespace jit
} // namespace torch
