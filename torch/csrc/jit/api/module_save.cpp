#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/unpickler.h>

namespace c10 {
// std::string serializeType(const Type &t);
TypePtr parseType(const std::string& pythonStr);
} // namespace c10

namespace torch {
namespace jit {

void Module::save(std::ostream& out, const ExtraFilesMap& extra_files) const {
  ExportModule(*this, out, extra_files, false /* bytecode_format */);
}

void Module::save(const std::string& filename, const ExtraFilesMap& extra_files)
    const {
  ExportModule(*this, filename, extra_files, false /* bytecode_format */);
}

void Module::_save_for_mobile(
    std::ostream& out,
    const ExtraFilesMap& extra_files,
    bool save_mobile_debug_info) const {
  ExportModule(
      *this,
      out,
      extra_files,
      true /* bytecode_format */,
      save_mobile_debug_info);
}

void Module::_save_for_mobile(
    const std::string& filename,
    const ExtraFilesMap& extra_files,
    bool save_mobile_debug_info) const {
  ExportModule(
      *this,
      filename,
      extra_files,
      true /* bytecode_format */,
      save_mobile_debug_info);
}

TypePtr resolveTypeName(
    const c10::QualifiedName& qn,
    std::shared_ptr<CompilationUnit> compilation_unit) {
  // HACK: first we check whether the name starts with special prefix to
  // tell if it's a supported pytorch class type. There are two special
  // prefixes. "__torch__" for nn module, and "torch.jit" from to_backend.
  // This is a reliable
  // check today, but there is no guarantee that this is the case. The
  // real solution is to merge type parsers so we can share class
  // resolution logic.
  static const c10::QualifiedName torchPrefix = "__torch__";
  static const c10::QualifiedName jitPrefix = "torch.jit";
  if (torchPrefix.isPrefixOf(qn) || jitPrefix.isPrefixOf(qn)) {
    if (compilation_unit->get_class(qn) == nullptr) {
      auto typeptr = ClassType::create(qn, compilation_unit, true);
      compilation_unit->register_type(typeptr);
    }
    return compilation_unit->get_class(qn);
  } else {
    return c10::parseType(qn.qualifiedName());
  }
}

c10::IValue readArchive(
    const std::string& archive_name,
    std::shared_ptr<mobile::CompilationUnit> compilation_unit,
    std::unique_ptr<caffe2::serialize::PyTorchStreamReader>& stream_reader) {
  std::stringstream picklename;
  picklename << archive_name << ".pkl";
  at::DataPtr pickle_ptr;
  size_t pickle_size;
  std::tie(pickle_ptr, pickle_size) =
      stream_reader->getRecord(picklename.str());

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

  TypeResolver type_resolver;

  auto obj_loader = [&](at::StrongTypePtr type, IValue input) {
    auto cls = type.type_->expect<at::ClassType>();
    auto qn = cls->name();
    c10::QualifiedName method_name(qn.value(), "__setstate__");
    auto setstate = compilation_unit->find_function(method_name);
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
      for (size_t i = 0; i < ndict; ++i) {
        std::stringstream name;
        name << it->key();
        cls->addOrCheckAttribute(name.str(), it->key().type());
        obj->setSlot(i, it->value());
        ++it;
      }
      return obj;
    }
  };

  static const std::string slash = "/";
  auto read_record = [&](const std::string& name) {
    std::size_t found = name.find(slash);
    std::stringstream ss;
    // In version 4, the tensor root_key doesn't include the parent path
    // To support backward compatibility, when the name doesn't include slash
    // assume it's version 4 and attach the archive_name_plus_slash
    // The example tensor format is:
    // torch._utils._rebuild_tensor_v2(
    //     pers.obj(('storage', torch.FloatStorage, '17', 'cpu', 22736),),
    //     0,
    //     (1, 464, 7, 7),
    //     (22736, 49, 7, 1),
    //     False,
    //     collections.OrderedDict())
    if (found == std::string::npos) {
      ss << archive_name << slash << name;
      return std::get<0>(stream_reader->getRecord(ss.str()));
    }

    // In version 4+, the tensor root_key in bytecode will include the parent
    // path. The example tensor format is: torch._utils._rebuild_tensor_v2(
    //     pers.obj(('storage', torch.FloatStorage, 'constants/17', 'cpu',
    //     22736),), 0, (1, 464, 7, 7), (22736, 49, 7, 1), False,
    //     collections.OrderedDict())
    ss << name;
    return std::get<0>(stream_reader->getRecord(ss.str()));
  };
  c10::optional<at::Device> device;

  Unpickler unpickler(
      reader,
      std::move(type_resolver),
      std::move(obj_loader),
      std::move(read_record),
      device);
  return unpickler.parse_ivalue();
}

void Module::_backport_for_mobile(
    const std::string& input_filename,
    const std::string& output_filename,
    const ExtraFilesMap& extra_files,
    bool save_mobile_debug_info) const {
  std::unique_ptr<FileAdapter> rai =
      std::make_unique<FileAdapter>(input_filename);
  auto reader = torch::make_unique<caffe2::serialize::PyTorchStreamReader>(
      std::move(rai));
  c10::optional<std::vector<IValue>> bvals;
  auto mcu = std::make_shared<mobile::CompilationUnit>();

  if (reader->hasRecord("bytecode.pkl")) {
    bvals = readArchive("bytecode", mcu, reader).toTuple()->elements();
  }
  if (bvals.has_value() && bvals.value().size() > 0 &&
      bvals.value()[0].isInt()) {
    uint64_t input_bytecode_version = bvals.value()[0].toInt();
    BackPortByteCode(
        *this,
        output_filename,
        input_bytecode_version,
        extra_files,
        true /* bytecode_format */,
        save_mobile_debug_info);
  } else {
    TORCH_WARN(
        "Fail to export bytecode model. Likely due to bytecode reading failure.");
  }
}

} // namespace jit
} // namespace torch
