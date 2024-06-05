#include <torch/csrc/jit/mobile/import_data.h>

#include <ATen/Functions.h>
#include <ATen/core/ivalue.h>
#include <c10/util/irange.h>
#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/mobile/file_format.h>
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/import_export_common.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/observer.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/unpickler.h>
#include <torch/custom_class.h>

#include <caffe2/serialize/in_memory_adapter.h>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

namespace torch {
namespace jit {
using caffe2::serialize::FileAdapter;
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::MemoryReadAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::ReadAdapterInterface;

namespace {

/**
 * Given a ZIP file containing a file named "data.pkl", uses Pickle to
 * deserialize the file and returns the IValue inside it.
 */
class IValueUnpickler final {
 public:
  explicit IValueUnpickler(std::unique_ptr<PyTorchStreamReader> reader);
  c10::IValue deserialize(std::optional<at::Device> device);

 private:
  c10::IValue readArchive(
      const std::string& archive_name,
      std::shared_ptr<mobile::CompilationUnit> mcu,
      std::optional<at::Device> device);

  std::shared_ptr<CompilationUnit> compilation_unit_;
  std::unique_ptr<PyTorchStreamReader> reader_;
};

IValueUnpickler::IValueUnpickler(std::unique_ptr<PyTorchStreamReader> reader)
    : compilation_unit_(std::make_shared<CompilationUnit>()),
      reader_(std::move(reader)) {}

c10::IValue IValueUnpickler::deserialize(std::optional<at::Device> device) {
  auto mcu = std::make_shared<mobile::CompilationUnit>();

  // NOLINTNEXTLINE(performance-move-const-arg)
  return readArchive("data", mcu, std::move(device));
}

c10::IValue IValueUnpickler::readArchive(
    const std::string& archive_name,
    std::shared_ptr<mobile::CompilationUnit> mcu,
    std::optional<at::Device> device) {
  std::stringstream picklename;
  picklename << archive_name << ".pkl";
  at::DataPtr pickle_ptr;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  size_t pickle_size;
  std::tie(pickle_ptr, pickle_size) = reader_->getRecord(picklename.str());

  size_t bytes_read = 0;
  auto data = reinterpret_cast<const char*>(pickle_ptr.get());
  auto reader = [&](char* buffer, size_t len) -> size_t {
    if (bytes_read >= pickle_size) {
      return 0;
    }
    len = std::min(pickle_size - bytes_read, len);
    // Copy len bytes into buffer
    const char* start = data + bytes_read;
    std::memcpy(buffer, start, len);
    bytes_read += len;
    return len;
  };

  static const c10::QualifiedName torchPrefix = "__torch__";
  auto type_resolver = [&](const c10::QualifiedName& qn) {
    TypePtr type;
    // HACK: first we check whether the name starts with `__torch__` to tell if
    // it's "supposed" to be a class type. This is a reliable check today, but
    // there is no guarantee that this is the case. The real solution is to
    // merge type parsers so we can share class resolution logic.
    if (torchPrefix.isPrefixOf(qn)) {
      if (compilation_unit_->get_class(qn) == nullptr) {
        auto typeptr = ClassType::create(qn, compilation_unit_, true);
        compilation_unit_->register_type(typeptr);
      }
      type = compilation_unit_->get_class(qn);
    } else {
      type = c10::parseType(qn.qualifiedName());
    }
    return c10::StrongTypePtr(compilation_unit_, type);
  };

  auto obj_loader = [&](const at::StrongTypePtr& type, IValue input) {
    auto cls = type.type_->expect<at::ClassType>();
    auto qn = cls->name();
    c10::QualifiedName method_name(qn.value(), "__setstate__");
    auto setstate = mcu->find_function(method_name);
    auto find_custom_class_with_setstate = [&qn]() -> c10::ClassTypePtr {
      auto custom_class_type = torch::jit::getCustomClass(qn->qualifiedName());
      if (custom_class_type && custom_class_type->findMethod("__setstate__")) {
        return custom_class_type;
      }
      return nullptr;
    };
    if (setstate) {
      auto obj = c10::ivalue::Object::create(type, 0);
      Stack stack({obj, input});
      setstate->run(stack);
      return obj;
    } else if (auto custom_class_type = find_custom_class_with_setstate()) {
      auto obj = c10::ivalue::Object::create(
          c10::StrongTypePtr(nullptr, custom_class_type), 1);
      Stack stack({obj, input});
      custom_class_type->getMethod("__setstate__").run(stack);
      return obj;
    } else {
      auto dict = std::move(input).toGenericDict();
      size_t ndict = dict.size();
      auto obj = c10::ivalue::Object::create(type, ndict);
      auto it = dict.begin();
      for (const auto i : c10::irange(ndict)) {
        std::stringstream name;
        name << it->key();
        cls->addOrCheckAttribute(name.str(), it->key().type());
        obj->setSlot(i, it->value());
        ++it;
      }
      return obj;
    }
  };

  auto read_record = [&](const std::string& name) {
    std::stringstream ss;
    ss << archive_name << "/" << name;
    return std::get<0>(reader_->getRecord(ss.str()));
  };

  Unpickler unpickler(
      reader,
      std::move(type_resolver),
      std::move(obj_loader),
      std::move(read_record),
      // NOLINTNEXTLINE(performance-move-const-arg)
      std::move(device),
      false,
      nullptr);
  return unpickler.parse_ivalue();
}

/**
 * Extracts and returns the parameter map serialized as ZIP + Pickle in @p rai.
 */
std::map<std::string, at::Tensor> load_parameters_from_zip(
    std::unique_ptr<ReadAdapterInterface> rai,
    std::optional<c10::Device> device) {
  auto reader = std::make_unique<PyTorchStreamReader>(std::move(rai));
  IValueUnpickler unpickler(std::move(reader));
  auto result = unpickler.deserialize(device).toGenericDict();
  std::map<std::string, at::Tensor> map;
  for (const auto& e : result) {
    auto key = e.key().toStringRef();
    auto value = e.value().toTensor().tensor_data();
    map[key] = value;
  }
  return map;
}

} // namespace

/**
 * Extracts the parameter map stored in @p module. Expects a layout
 * compatible with the one created by #_save_parameters().
 */
std::map<std::string, at::Tensor> mobile_module_to_parameter_map(
    const mobile::Module& module) {
  // Safely look for a slot with the expected name. Note that
  // c10::ivalue::Object::getAttr() is not safe if the attribute isn't present.
  auto obj = module._ivalue();
  const std::vector<IValue>& slots = obj->slots();
  for (const auto i : c10::irange(slots.size())) {
    if (obj->type()->getAttributeName(i) ==
        mobile::internal::kSavedParametersAttributeName) {
      // Found a slot with the right name; make sure it's a
      // Dict<string, Tensor>.
      c10::IValue data = slots[i];
      if (data.isGenericDict()) {
        auto data_dict = data.toGenericDict();

        // The key and value should be DynamicTypes that wrap String and Tensor.
        c10::DynamicType* keyType =
            data_dict.keyType()->castRaw<c10::DynamicType>();
        c10::DynamicType* valueType =
            data_dict.valueType()->castRaw<c10::DynamicType>();
        if (keyType != nullptr &&
            keyType->fallback()->kind() == TypeKind::StringType &&
            valueType != nullptr &&
            valueType->fallback()->kind() == TypeKind::TensorType) {
          // Name and type are good; copy the contents to the output map.
          std::map<std::string, at::Tensor> params;
          for (const auto& e : data_dict) {
            // The source Tensor points into the flatbuffer data associated with
            // the Module. But, this Tensor needs to outlive the Module, since
            // the caller of _load_parameters() won't have a pointer to the
            // Module. So, return a deep copy.
            const auto& source = e.value().toTensor();
            at::Tensor copy = at::empty_like(source); // Must be the same shape.
            copy.copy_(source);

            params[e.key().toStringRef()] = copy;
          }
          return params;
        }
      }
    }
  }

  TORCH_CHECK(
      false,
      "Could not find Dict<string, Tensor> named '",
      mobile::internal::kSavedParametersAttributeName,
      "' in deserialized mobile::Module");
}

static std::map<std::string, at::Tensor> _load_parameters_bytes(
    std::shared_ptr<char> data,
    size_t size,
    std::optional<at::Device> device) {
  TORCH_CHECK(size >= kFileFormatHeaderSize, "Unrecognized data format");
  FileFormat format = getFileFormat(data.get());
  // Call the appropriate parser.
  std::map<std::string, at::Tensor> map;
  switch (format) {
    case FileFormat::FlatbufferFileFormat: {
      auto m = parse_flatbuffer_no_object(data, size, device);
      map = mobile_module_to_parameter_map(m);
      break;
    }

    case FileFormat::ZipFileFormat: {
      auto rai = std::make_unique<caffe2::serialize::MemoryReadAdapter>(
          data.get(), size);
      map = load_parameters_from_zip(std::move(rai), device);
      break;
    }

    default:
      TORCH_CHECK(false, "Unrecognized data format");
  }
  return map;
}

std::map<std::string, at::Tensor> _load_parameters(
    std::istream& in,
    std::optional<at::Device> device) {
  auto [data, size] = get_stream_content(in);
  return _load_parameters_bytes(std::move(data), size, device);
}

std::map<std::string, at::Tensor> _load_parameters(
    const std::string& filename,
    std::optional<at::Device> device) {
  auto [data, size] = get_file_content(filename.c_str());
  return _load_parameters_bytes(std::move(data), size, device);
}

} // namespace jit
} // namespace torch
