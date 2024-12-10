#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/parse_bytecode.h>
#include <torch/csrc/jit/mobile/parse_operators.h>

#include <ATen/core/ivalue.h>
#include <ATen/core/qualified_name.h>
#include <c10/util/Exception.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/irange.h>
#include <caffe2/serialize/in_memory_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <caffe2/serialize/istream_adapter.h>
#include <caffe2/serialize/versions.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/mobile/file_format.h>
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
#include <torch/csrc/jit/mobile/observer.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/mobile/upgrader_mobile.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/import_export_constants.h>
#include <torch/csrc/jit/serialization/import_export_functions.h>
#include <torch/custom_class.h>
#include <optional>
#include <string>
#include <vector>

// The import process to serialize the bytecode package.
// An example for bytecode.pkl of a small mobile_module looks like:
// (4,  # model version number (caffe2::serialize::kProducedBytecodeVersion)
//  # first method
//  (
//   # function name
//   '__torch__.m.forward',
//   # code
//   (('instructions',
//     (('STOREN', 1, 2),
//      ('DROPR', 1, 0),
//      ('MOVE', 2, 0),
//      ('OP', 0, 0),
//      ('RET', 0, 0))),
//    ('operators', (('aten::Int', 'Tensor'),)),
//    ('constants', ()),
//    ('types', ()),
//    ('register_size', 2)),
//   # schema -- optional (forward-compatible addition to version 4)
//   (('arguments',
//     ((('name', 'x'), ('type', 'Tensor'), ('default_value', 13)),
//      ...)),  # more args follow here
//    ('returns',
//     ((('name', ''), ('type', 'Tensor'), ('default_value', None)),
//      ...)),  # more return values follow here
//   )),
//  # more methods follow here
//  ...)

// In addition, the module debugging information can be saved
// in mobile_debug_handles.pkl. An example for it looks like:
// (4,
//  ('__torch__.m.forward',
//   (('module_debug_handles', 10))))
//   Here 10 is the debug handle.
// We also store separately and optionally callstack_debug_map.
// This serializes inlined callstack (InlinedCallStack data structure)
// corresponding to the debug handles.
// Callstack_debug_map serializes tuples of
// (int64_t(debug_handle), int64_t(source_range_tag), InlinedCallStack)
// source_range_tag maps to .debug_pkl files where this tag maps it to
// source range.
// InlinedCallStack is serialized as:
// IValue(InlinedCallStack) = {IValue(ModuleInstanceInfo),
// int64_t(source_range_tag), IValue(InlinedCallStack)} ModuleInstanceInfo is
// serialized as a tuple of (class_type_name, instance_name)

// Note that currently the backward compatibility is not supported by bytecode.
// This format and process need to be revisited and redesigned if we want to
// support backward compatibility in future.

// Note that the following function-schema fields are not supported:
//  - Argument::{known_length_,kwarg_only_}
//  - FunctionSchema::{overload_name_, is_vararg_, is_varret_}

namespace torch::jit {
using caffe2::serialize::MemoryReadAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::ReadAdapterInterface;

OpCode parseOpCode(const char* str);

TypePtr resolveTypeNameMobile(
    const c10::QualifiedName& qn,
    const std::shared_ptr<CompilationUnit>& compilation_unit) {
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

c10::StrongTypePtr typeResolverMobile(
    const c10::QualifiedName& qn,
    const std::shared_ptr<CompilationUnit>& compilation_unit) {
  return c10::StrongTypePtr(
      compilation_unit, resolveTypeNameMobile(qn, compilation_unit));
}

c10::intrusive_ptr<c10::ivalue::Object> objLoaderMobile(
    const at::StrongTypePtr& type,
    const IValue& input,
    mobile::CompilationUnit& mobile_compilation_unit) {
  auto cls = type.type_->expect<at::ClassType>();
  auto qn = cls->name();
  c10::QualifiedName method_name(qn.value(), "__setstate__");
  auto setstate = mobile_compilation_unit.find_function(method_name);
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
    auto dict = input.toGenericDict();
    size_t ndict = dict.size();
    auto obj = c10::ivalue::Object::create(type, ndict);
    auto it = dict.begin();
    for (const auto i : c10::irange(ndict)) {
      cls->addOrCheckAttribute(it->key().toStringRef(), it->key().type());
      obj->setSlot(i, it->value());
      ++it;
    }
    return obj;
  }
}

bool isTensorInBytecodeArchive(
    caffe2::serialize::PyTorchStreamReader& stream_reader) {
  auto records = stream_reader.getAllRecords();
  for (const auto& record : records) {
    if (record.find("bytecode/") != std::string::npos) {
      return true;
    }
  }
  return false;
}

namespace {

void tryRegisterMethod(const std::vector<c10::Argument>& args, Function& func) {
  if (args.empty() || args[0].name() != "self") {
    return;
  }

  if (auto cls = args[0].type()->castRaw<ClassType>()) {
    if (C10_UNLIKELY(cls->findMethod(func.name()))) {
      return;
    }
    cls->addMethod(&func);
  }
}

// The deserializer class which loads the bytecode package from bc files.
class BytecodeDeserializer final {
 public:
  explicit BytecodeDeserializer(
      std::unique_ptr<PyTorchStreamReader> reader,
      uint64_t module_load_options = 0);
  mobile::Module deserialize(std::optional<at::Device> device);
  mobile::Module deserialize(
      std::optional<at::Device> device,
      ExtraFilesMap& extra_files);
  void deserialize_only_extra(
      std::optional<at::Device> device,
      ExtraFilesMap& extra_files);

 private:
  TypePtr resolveTypeName(const c10::QualifiedName& qn);
  void init_upgrader(mobile::Function* function);
  void parseMethods(
      c10::ivalue::TupleElements&& vals,
      std::optional<c10::ivalue::TupleElements>&& debug_handles,
      mobile::CompilationUnit& mcu);
  c10::IValue readArchive(
      const std::string& archive_name,
      std::shared_ptr<mobile::CompilationUnit> mcu);
  void parseFunctionSchema(
      const std::string& function_name,
      IValue* schemaTable,
      const int64_t& model_version,
      mobile::Function* function);
  std::shared_ptr<CompilationUnit> compilation_unit_;
  std::unordered_set<std::string> imported_libs_;
  std::unique_ptr<PyTorchStreamReader> reader_{};
  std::optional<at::Device> device_;
  uint64_t module_load_options_;
  // From `version` or `.data/version` in model.ptl and it's compute
  // dynamically. It's used for finding the minimum required runtime to run all
  // operators from the given model. If it's less than the current runtime,
  // upgrader will be applied at loading stage.
  uint64_t operator_version_{0};
  uint64_t bytecode_version_{0};
};

BytecodeDeserializer::BytecodeDeserializer(
    std::unique_ptr<PyTorchStreamReader> reader,
    uint64_t module_load_options)
    : compilation_unit_(std::make_shared<CompilationUnit>()),
      reader_(std::move(reader)),
      module_load_options_(module_load_options) {}

TypePtr BytecodeDeserializer::resolveTypeName(const c10::QualifiedName& qn) {
  return resolveTypeNameMobile(qn, compilation_unit_);
}

// It requires compilation_unit_ when parsing function schema. Keep it in
// BytecodeDeserializer. It may be refacotred later to make it independent
// of the specific BytecodeDeserializer, like parsing other tables
void BytecodeDeserializer::parseFunctionSchema(
    const std::string& function_name,
    IValue* schemaTable,
    const int64_t& model_version,
    mobile::Function* function) {
  // function schema
  if (schemaTable) { // (schema is optional for back compat)
    auto parseArgList = [this,
                         function](c10::ivalue::TupleElements&& argTables) {
      std::vector<c10::Argument> args;
      for (auto& argTable : argTables) {
        auto argTableElements = std::move(argTable.toTupleRef()).elements();
        auto name =
            expect_field(argTableElements, "name", BYTECODE_INDEX_ARGUMENT_NAME)
                .toStringRef();
        c10::TypePtr type = resolveTypeName(
            (expect_field(
                 argTableElements, "type", BYTECODE_INDEX_ARGUMENT_TYPE))
                .toStringRef());
        IValue default_value = expect_field(
            argTableElements,
            "default_value",
            BYTECODE_INDEX_ARGUMENT_DEFAULT_VALUE);
        args.emplace_back(
            name,
            std::move(type),
            std::nullopt /*N*/,
            std::move(default_value));
      }
      tryRegisterMethod(args, *function);
      return args;
    };
    auto schemaTableElements = std::move(schemaTable->toTupleRef()).elements();
    auto arg_list = std::move(expect_field(
                                  schemaTableElements,
                                  "arguments",
                                  BYTECODE_INDEX_SCHEMA_ARGUMENTS)
                                  .toTupleRef())
                        .elements();
    auto ret_list =
        std::move(
            expect_field(
                schemaTableElements, "returns", BYTECODE_INDEX_SCHEMA_RETURNS)
                .toTupleRef())
            .elements();
    c10::FunctionSchema schema(
        function_name,
        "" /*overload_name*/,
        parseArgList(std::move(arg_list)),
        parseArgList(std::move(ret_list)),
        false /*is_varargs*/,
        false /*is_varret*/);
    function->setSchema(std::move(schema));
  }
}

void BytecodeDeserializer::init_upgrader(mobile::Function* function) {
  for (auto& byteCodeFunctionWithOperator : getUpgraderBytecodeList()) {
    function->append_function(byteCodeFunctionWithOperator.function);
  }
}

void BytecodeDeserializer::parseMethods(
    c10::ivalue::TupleElements&& vals,
    std::optional<c10::ivalue::TupleElements>&& debug_handles,
    mobile::CompilationUnit& mcu) {
  TORCH_CHECK(!vals.empty(), "Bytecode has no elements. ");
  // Initialized with the version number when kProducedBytecodeVersion was
  // introduced. The old models (some of them already in production) without
  // version number are seen as version 3 (deprecated).
  constexpr uint64_t default_version = 0x3L;
  bytecode_version_ = default_version;
  size_t method_i_start = 0;
  if (vals[0].isInt()) {
    bytecode_version_ = vals[0].toInt();
    method_i_start = 1;
  }
  TORCH_CHECK(
      caffe2::serialize::kMinSupportedBytecodeVersion <= bytecode_version_ &&
          bytecode_version_ <= caffe2::serialize::kMaxSupportedBytecodeVersion,
      "Lite Interpreter version number does not match. ",
      "The model version must be between ",
      caffe2::serialize::kMinSupportedBytecodeVersion,
      " and ",
      caffe2::serialize::kMaxSupportedBytecodeVersion,
      " but the model version is ",
      bytecode_version_);

  if (debug_handles) {
    TORCH_CHECK(
        debug_handles->size() == vals.size(),
        "The numbers of bytecode values and debug info values do not match.");
  }

  // Process all methods in this mobile module.
  for (const auto i : c10::irange(method_i_start, vals.size())) {
    auto element = std::move(vals[i]);
    auto m_tuple = std::move(element.toTupleRef()).elements();
    const std::string& function_name = m_tuple[0].toStringRef();
    auto codeTableElements = std::move(m_tuple[1].toTupleRef()).elements();
    IValue* schemaTable = // older files do not store function schema
        (bytecode_version_ > 0x4L ||
         (bytecode_version_ == 0x4L && m_tuple.size() >= 3))
        ? &m_tuple[2]
        : nullptr;
    auto function =
        std::make_unique<mobile::Function>(c10::QualifiedName(function_name));

    auto ins_list =
        std::move(
            expect_field(
                codeTableElements, "instructions", BYTECODE_INDEX_INSTRUCTION)
                .toTupleRef())
            .elements();
    auto ops_list =
        std::move(expect_field(
                      codeTableElements, "operators", BYTECODE_INDEX_OPERATOR)
                      .toTupleRef())
            .elements();
    auto consts_list =
        std::move(expect_field(
                      codeTableElements, "constants", BYTECODE_INDEX_CONSTANT)
                      .toTupleRef())
            .elements();
    auto types_list =
        std::move(expect_field(codeTableElements, "types", BYTECODE_INDEX_TYPE)
                      .toTupleRef())
            .elements();
    int64_t register_size =
        expect_field(
            codeTableElements, "register_size", BYTECODE_INDEX_REGISTER_SIZE)
            .toInt();

    c10::ivalue::TupleElements debug_handles_m_tuple;
    if (debug_handles) {
      debug_handles_m_tuple =
          std::move(std::move((*debug_handles)[i]).toTupleRef()).elements();
    }
    init_upgrader(function.get());
    // 1. First pass all operators from models
    parseOperators(std::move(ops_list), module_load_options_, function.get());

    // 2. Decides if upgrader is needed
    bool use_upgrader =
        (operator_version_ < caffe2::serialize::kProducedFileFormatVersion);

    parseInstructions(
        function_name,
        std::move(ins_list),
        debug_handles_m_tuple,
        function.get());

    // 3. If upgrader is needed, change change the OP instrunction to CALL
    // instruction (In next PR, use_upgrader will be parsed to parseInstruction
    // function and do the actual change)
    if (use_upgrader) {
      applyUpgrader(function.get(), operator_version_);
    }

    parseConstants(consts_list, function.get());

    parseTypes(types_list, function.get());

    function->set_register_size(register_size);

    parseFunctionSchema(
        function_name, schemaTable, bytecode_version_, function.get());

    mcu.register_function(std::move(function));
  }
}

void BytecodeDeserializer::deserialize_only_extra(
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  device_ = device;
  for (const auto& kv : extra_files) {
    const std::string& key = "extra/" + kv.first;
    if (reader_->hasRecord(key)) {
      auto [meta_ptr, meta_size] = reader_->getRecord(key);
      extra_files[kv.first] =
          std::string(static_cast<char*>(meta_ptr.get()), meta_size);
    }
  }
}

mobile::Module BytecodeDeserializer::deserialize(
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  deserialize_only_extra(device, extra_files);
  return deserialize(device);
}

mobile::Module BytecodeDeserializer::deserialize(
    std::optional<at::Device> device) {
  device_ = device;
  auto mcu = std::make_shared<mobile::CompilationUnit>();

  // bvals can have 2 possible formats:
  //
  // 1. Old format: bvals is an array (Tuple) of N elements, each element being
  // itself a Tuple(method_name, method_table).
  //
  // 2. New format: bvals is an array (Tuple) of 1+N elements. The first element
  // being a Tuple (int, table), and the integer stands for the bytecode version
  // number. The rest of the elements are the same as before.
  //
  auto bvals = std::move(readArchive("bytecode", mcu).toTupleRef()).elements();

  std::optional<c10::ivalue::TupleElements> debug_handles;
  bool has_debug_handles{false};
  if (reader_->hasRecord("mobile_debug_handles.pkl")) {
    debug_handles =
        std::move(readArchive("mobile_debug_handles", mcu).toTupleRef())
            .elements();
    has_debug_handles = true;
  }
  operator_version_ = reader_->version();
  parseMethods(std::move(bvals), std::move(debug_handles), *mcu);
  auto m = mobile::Module(readArchive("data", mcu).toObject(), mcu);
  m.set_min_operator_version(operator_version_);
  m.set_bytecode_version(bytecode_version_);
  m.setHasDebugHandles(has_debug_handles);
#if defined(SYMBOLICATE_MOBILE_DEBUG_HANDLE)
  MobileDebugTable debug_table = MobileDebugTable(reader_, compilation_unit_);
  m.setDebugTable(std::move(debug_table));
#endif
  return m;
}

c10::IValue BytecodeDeserializer::readArchive(
    const std::string& archive_name,
    std::shared_ptr<mobile::CompilationUnit> mcu) {
  auto type_resolver = [this](const c10::QualifiedName& qn) {
    return typeResolverMobile(qn, compilation_unit_);
  };

  auto obj_loader = [&](const at::StrongTypePtr& type, const IValue& input) {
    return objLoaderMobile(type, input, *mcu);
  };

  bool bytecode_tensor_in_constants_archive =
      (archive_name == "bytecode" && !isTensorInBytecodeArchive(*reader_));

  auto ivalues = torch::jit::readArchiveAndTensors(
      archive_name,
      /*pickle_prefix=*/"",
      /*tensor_prefix=*/
      bytecode_tensor_in_constants_archive ? "constants/" : "",
      type_resolver,
      obj_loader,
      device_,
      *reader_,
      nullptr);
  return ivalues;
}

mobile::Module _load_for_mobile_impl(
    std::unique_ptr<ReadAdapterInterface> rai,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    uint64_t module_load_options) {
  auto observer = torch::observerConfig().getModuleObserver();
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  auto instance_key = std::rand();

  std::unordered_map<std::string, std::string> metadata_map;
  if (observer) {
    observer->onEnterLoadModel(instance_key);
    auto defaultExtraFileList = observer->getDefaultExtraFiles();
    // Add files in defaultExtraFileList to fail_extra_files and extra_files
    for (const auto& fileName : defaultExtraFileList) {
      extra_files.insert(std::make_pair(fileName, ""));
    }
  }

  const size_t model_size = rai != nullptr ? rai->size() : 0;
  auto reader = std::make_unique<PyTorchStreamReader>(std::move(rai));
  if (module_load_options &
      MobileModuleLoadOptions::PARSE_ALL_EXTRA_FILE_MAPS) {
    // ExtraFilesMap is serialized with a "extra/", hence it is necessary to
    // account for when we de-serialize de-serialized filemap key values contain
    // prefix and we need to remove prior to construct the map. "extra/" string
    // has a length of 6 characters, hence we need only sub-string 6th position
    // of a string. Please refer to following link for a detail:
    // https://www.internalfb.com/code/fbsource/[9996fcb7a6fb]/fbcode/caffe2/torch/csrc/jit/mobile/import.cpp?lines=427-434
    std::vector<std::string> all_files = reader->getAllRecords();
    for (auto& file_name : all_files) {
      if (file_name.find("extra/") == 0) {
        extra_files[file_name.substr(6)] = "";
      }
    }
  }
  BytecodeDeserializer deserializer(std::move(reader), module_load_options);

  std::string error_message;
  auto guard = c10::make_scope_exit([&]() {
    if (!observer) {
      return;
    }
    deserializer.deserialize_only_extra(device, extra_files);

    metadata_map = observer->processMetadataFromExtra(extra_files);

    observer->onFailLoadModel(
        instance_key,
        error_message.empty() ? "Unknown exception" : error_message.c_str(),
        metadata_map);
  });

  try {
    mobile::Module result = deserializer.deserialize(device, extra_files);
    if (observer) {
      // Add model_name and model_size to metadata_map
      extra_files.insert(std::make_pair("model_name", result.name()));
      extra_files.insert(
          std::make_pair("model_size", std::to_string(model_size)));
      metadata_map = observer->processMetadataFromExtra(extra_files);
      observer->onExitLoadModel(instance_key, metadata_map);
    }
    result.setMetadata(metadata_map);
    guard.release();
    return result;
  } catch (c10::Error& error) {
    error_message = error.what();
    TORCH_RETHROW(error);
  }
}

mobile::Module _load_mobile_from_bytes(
    const std::shared_ptr<char>& data,
    size_t size,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    uint64_t module_load_options) {
  TORCH_CHECK(size >= kFileFormatHeaderSize, "Format error");
  auto format = getFileFormat(data.get());
  switch (format) {
    case FileFormat::ZipFileFormat: {
      std::unique_ptr<ReadAdapterInterface> rai =
          std::make_unique<MemoryReadAdapter>(data.get(), size);
      return _load_for_mobile_impl(
          std::move(rai), device, extra_files, module_load_options);
    }
    case FileFormat::FlatbufferFileFormat: {
      return parse_and_initialize_mobile_module(
          data, size, device, &extra_files);
    }
    default: {
      TORCH_CHECK(false, "Format error");
    }
  }
}

} // namespace

mobile::Module _load_for_mobile(
    std::istream& in,
    std::optional<at::Device> device) {
  ExtraFilesMap extra_files;
  return _load_for_mobile(in, device, extra_files);
}

mobile::Module _load_for_mobile(
    const std::string& filename,
    std::optional<at::Device> device) {
  ExtraFilesMap extra_files;
  return _load_for_mobile(filename, device, extra_files);
}

mobile::Module _load_for_mobile(
    std::unique_ptr<ReadAdapterInterface> rai,
    std::optional<c10::Device> device) {
  ExtraFilesMap extra_files;
  return _load_for_mobile(std::move(rai), device, extra_files);
}

mobile::Module _load_for_mobile(
    std::istream& in,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    uint64_t module_load_options) {
  if (getFileFormat(in) == FileFormat::FlatbufferFileFormat) {
    auto [data, size] = get_stream_content(in);
    return _load_mobile_from_bytes(
        data, size, device, extra_files, module_load_options);
  }
  auto rai = std::make_unique<caffe2::serialize::IStreamAdapter>(&in);
  auto module = _load_for_mobile_impl(
      std::move(rai), device, extra_files, module_load_options);
  return module;
}

mobile::Module _load_for_mobile(
    const std::string& filename,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  return _load_for_mobile(
      filename, device, extra_files, kDefaultMobileLoadOptions);
}

mobile::Module _load_for_mobile(
    const std::string& filename,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    uint64_t module_load_options) {
  auto observer = torch::observerConfig().getModuleObserver();
  if (observer) {
    extra_files.insert(std::make_pair("model_path", filename));
  }
  auto format = getFileFormat(filename);

  if (format == FileFormat::FlatbufferFileFormat) {
    auto [data, size] = get_file_content(filename.c_str());
    return _load_mobile_from_bytes(
        data, size, device, extra_files, module_load_options);
  }

  auto rai = std::make_unique<caffe2::serialize::FileAdapter>(filename);
  return _load_for_mobile_impl(
      std::move(rai), device, extra_files, module_load_options);
}

TORCH_API mobile::Module _load_for_mobile(
    std::unique_ptr<ReadAdapterInterface> rai,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    uint64_t module_load_options) {
  // TODO optimize file read for non-flatbuffer models
  auto [data, size] = get_rai_content(rai.get());
  return _load_mobile_from_bytes(
      data, size, device, extra_files, module_load_options);
}

void _load_extra_only_for_mobile(
    const std::string& filename,
    std::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  auto observer = torch::observerConfig().getModuleObserver();
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  auto instance_key = std::rand();
  if (observer) {
    observer->onEnterLoadModel(instance_key);
  }

  auto format = getFileFormat(filename);
  switch (format) {
    case FileFormat::ZipFileFormat: {
      auto rai = std::make_unique<caffe2::serialize::FileAdapter>(filename);
      auto reader = std::make_unique<PyTorchStreamReader>(std::move(rai));
      BytecodeDeserializer deserializer(std::move(reader));
      deserializer.deserialize_only_extra(device, extra_files);
      break;
    }
    case FileFormat::FlatbufferFileFormat: {
      // TODO: the current flatbuffers implementation will always load the
      // whole module including the extra files. Ideally it should be
      // possible to just get the extra files given data
      load_mobile_module_from_file(filename, std::nullopt, &extra_files);
      break;
    }
    default: {
      TORCH_CHECK(false, "Format error");
    }
  }
}

namespace mobile {

std::set<std::string> _export_operator_list(
    torch::jit::mobile::Module& module) {
  std::set<std::string> operator_list;
  for (Method func : module.get_methods()) {
    const Function& function = func.function();
    const auto& code = function.get_code();
    // op_names below isn't a list of unique operator names. In fact
    // it can contain the same operator name many many times, so we need
    // to de-dup the list by adding all the operator names into
    // an std::set<std::string>.
    std::vector<c10::OperatorName> const& op_names = code.op_names_;
    for (auto& op_name : op_names) {
      operator_list.insert(toString(op_name));
    }
  }
  return operator_list;
}

} // namespace mobile
} // namespace torch::jit
