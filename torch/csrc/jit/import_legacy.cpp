#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/type_resolver_util.h>

#include <torch/csrc/jit/import_export_helpers.h>
#include <torch/csrc/jit/import_legacy.h>
#include <torch/csrc/jit/import_source.h>
#include <torch/csrc/jit/pickle.h>
#include <torch/csrc/jit/script/script_type_parser.h>
#include <torch/csrc/jit/source_range_serialization.h>
#include <torch/csrc/jit/source_range_serialization_impl.h>

#include "caffe2/core/common.h"
#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/proto/torch_pb.h"
#include "caffe2/serialize/inline_container.h"

#include <ATen/ATen.h>

namespace torch {
namespace jit {

using caffe2::serialize::PyTorchStreamReader;
void postSetStateValidate(const IValue& v);
namespace {

struct ClassResolver : public script::Resolver {
  explicit ClassResolver(script::SourceImporter source_importer)
      : source_importer_(std::move(source_importer)) {}
  TypePtr resolveType(const std::string& name, const SourceRange& loc)
      override {
    return source_importer_.loadNamedType(c10::QualifiedName(name));
  }

 private:
  script::SourceImporter source_importer_;
};

class ScriptModuleDeserializer final {
 public:
  ScriptModuleDeserializer(
      std::shared_ptr<script::CompilationUnit> cu,
      std::unique_ptr<PyTorchStreamReader> reader,
      const c10::optional<at::Device>& device)
      : compilation_unit_(cu),
        reader_(std::move(reader)),
        device_(device),
        source_importer_(
            compilation_unit_,
            &constants_table_,
            [this](const std::string& qualifier) {
              return findSourceInArchiveFromQualifier(
                  *reader_, export_prefix_, qualifier);
            },
            reader_->version()) {}

  script::Module LEGACY_deserialize();

 private:
  at::Tensor LEGACY_loadTensor(
      const torch::TensorDef& tensor_proto,
      std::unordered_map<std::string, at::Storage>& storageMap);
  void LEGACY_loadTensorTable(torch::ModelDef* model_def);
  void LEGACY_moduleSetState(const script::Module& module, IValue state);
  IValue LEGACY_loadPickleArchive(const std::string& name);
  script::Module LEGACY_convertModule(const torch::ModuleDef& module_def);

  std::vector<IValue> LEGACY_pickled_ivalues_;
  std::vector<std::string> LEGACY_moduleStack_;

  std::shared_ptr<Source> sourceLoader(const std::string& qualifier);

  std::shared_ptr<script::CompilationUnit> compilation_unit_;
  std::unique_ptr<PyTorchStreamReader> reader_;
  c10::optional<at::Device> device_;
  std::vector<at::Tensor> constants_table_;
  script::SourceImporter source_importer_;
  std::string export_prefix_ = "code/";
};

script::Module ScriptModuleDeserializer::LEGACY_deserialize() {
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
  auto proto_version = model_def.proto_version();
  export_prefix_ = "libs/";

  LEGACY_loadTensorTable(&model_def);
  AT_ASSERT(proto_version < 6);
  if (proto_version == 2) {
    const auto& list =
        LEGACY_loadPickleArchive("attributes.pkl").toGenericList();
    LEGACY_pickled_ivalues_.insert(
        LEGACY_pickled_ivalues_.end(), list.begin(), list.end());
  } else if (proto_version >= 3) {
    LEGACY_pickled_ivalues_ =
        LEGACY_loadPickleArchive("attributes.pkl").toTuple()->elements();
  }
  LEGACY_moduleStack_.push_back("__torch__");
  const auto& module_def = model_def.main_module();
  return LEGACY_convertModule(module_def);
}

IValue ScriptModuleDeserializer::LEGACY_loadPickleArchive(
    const std::string& name) {
  at::DataPtr attributes_ptr;
  size_t attributes_size;
  std::tie(attributes_ptr, attributes_size) = reader_->getRecord(name);
  auto ivalue = unpickle(
      reinterpret_cast<const char*>(attributes_ptr.get()),
      attributes_size,
      [&](const c10::QualifiedName& qn) {
        auto cls = source_importer_.loadNamedType(qn)->expect<ClassType>();
        return c10::StrongTypePtr(compilation_unit_, std::move(cls));
      },
      &constants_table_);
  return ivalue;
}

void ScriptModuleDeserializer::LEGACY_loadTensorTable(
    torch::ModelDef* model_def) {
  std::unordered_map<std::string, at::Storage> storageMap;
  for (const torch::TensorDef& tensor : model_def->tensors()) {
    constants_table_.emplace_back(LEGACY_loadTensor(tensor, storageMap));
  }
}

at::Tensor ScriptModuleDeserializer::LEGACY_loadTensor(
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

void ScriptModuleDeserializer::LEGACY_moduleSetState(
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
    setstate->run({module._ivalue()});
  } else if (setstate->num_inputs() == 2) {
    setstate->run({module._ivalue(), state});
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
    auto is_digits = [](const std::string& str) {
      return std::all_of(str.begin(), str.end(), ::isdigit);
    };
    auto sanitized = is_digits(atom) ? std::string("_") + atom : atom;
    LEGACY_moduleStack_.emplace_back(sanitized);
  }
  auto module = script::Module(
      c10::QualifiedName(LEGACY_moduleStack_), compilation_unit_);
  for (int i = 0; i < module_def.submodules_size(); ++i) {
    const torch::ModuleDef& sub_def = module_def.submodules(i);
    auto submodule = LEGACY_convertModule(sub_def);
    module.register_module(sub_def.name(), submodule);
  }
  for (int i = 0; i < module_def.parameters_size(); ++i) {
    const torch::ParameterDef& param_def = module_def.parameters(i);
    at::Tensor tensor = constants_table_.at(param_def.tensor_id());
    if (param_def.is_buffer()) {
      module.register_buffer(param_def.name(), tensor);
    } else {
      module.register_parameter(param_def.name(), tensor, /*is_buffer=*/false);
    }
  }
  script::ScriptTypeParser typeParser(
      std::make_shared<ClassResolver>(source_importer_));
  for (int i = 0; i < module_def.attributes_size(); ++i) {
    const torch::AttributeDef& attr_def = module_def.attributes(i);
    if (module.hasattr(attr_def.name())) {
      // this attribute was already registered as a buffer above.
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

    source_importer_.LEGACY_import_methods(module, src);
  }

  if (module_def.has_get_state_attribute_id()) {
    LEGACY_moduleSetState(
        module,
        LEGACY_pickled_ivalues_.at(module_def.get_state_attribute_id()));
  }

  const ClassTypePtr& module_type = module._ivalue()->type();
  for (size_t i = 0, N = module_type->numAttributes(); i < N; ++i) {
    // Verify that all the non-optional attributes have been initialized
    // TODO: Issue #20497
    const IValue& v = module._ivalue()->getSlot(i);
    if (module_type->getAttribute(i)->kind() != TypeKind::OptionalType) {
      TORCH_CHECK(
          !v.isNone(),
          "The field '",
          module_type->getAttributeName(i),
          "' was left unitialized after __setstate__, but expected a ",
          "value of type '",
          v.type()->python_str(),
          "'");
    }
  }

  for (size_t i = 0; i < numPushed; i++) {
    LEGACY_moduleStack_.pop_back();
  }
  return module;
}

} // namespace

script::Module LEGACY_deserialize(
    std::shared_ptr<script::CompilationUnit> cu,
    std::unique_ptr<caffe2::serialize::PyTorchStreamReader> reader,
    const c10::optional<c10::Device>& device) {
  ScriptModuleDeserializer deserializer(cu, std::move(reader), device);
  return deserializer.LEGACY_deserialize();
}

} // namespace jit
} // namespace torch
