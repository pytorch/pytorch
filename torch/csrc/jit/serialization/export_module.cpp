#include <torch/csrc/jit/serialization/export.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/backends/backend_debug_handler.h>
#include <torch/csrc/jit/backends/backend_debug_info.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/attributes.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/type_hashing.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/method.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/callstack_debug_info_serialization.h>
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <torch/csrc/jit/serialization/import_export_constants.h>
#include <torch/csrc/jit/serialization/import_export_functions.h>
#include <torch/csrc/jit/serialization/import_export_helpers.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>

#include <caffe2/serialize/inline_container.h>

#include <ATen/ATen.h>

#include <ATen/core/jit_type.h>
#include <ATen/core/qualified_name.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {

namespace {

ExportModuleExtraFilesHook& GetExtraFilesHook() {
  static ExportModuleExtraFilesHook func = nullptr;
  return func;
}

std::unordered_set<const FunctionSchema*> getInterfaceCalls(Graph& graph) {
  std::unordered_set<const FunctionSchema*> ret;
  auto nodes = findAllNodes(graph, c10::prim::CallMethod, true);
  for (Node* node : nodes) {
    if (auto iface = node->input(0)->type()->castRaw<InterfaceType>()) {
      ret.insert(iface->getMethod(node->s(attr::name)));
    }
  }
  return ret;
}

struct ModuleMethod {
  ModuleMethod(const Module& m, const GraphFunction& f, c10::QualifiedName n)
      : module(m), function(f), exportName(std::move(n)) {}
  const Module& module;
  const GraphFunction& function;
  c10::QualifiedName exportName;
};

std::vector<ModuleMethod> getModuleInterfaceExports(
    const Module& module,
    const std::unordered_set<const FunctionSchema*>& schemas) {
  if (schemas.size() == 0) {
    return {};
  }
  std::unordered_set<std::string> names;
  for (auto schema : schemas) {
    names.insert(schema->name());
  }
  std::vector<ModuleMethod> ret;
  for (const auto& submodule : module.modules()) {
    for (const auto& method : submodule.get_methods()) {
      const auto& f = toGraphFunction(method.function());
      if (names.find(f.qualname().name()) != names.end()) {
        ret.emplace_back(submodule, f, f.qualname());
      }
    }
  }
  return ret;
}

void exportFunction(
    BytecodeExportSet& exportSet,
    const ModuleMethod& method,
    bool toplevel = false) {
  const auto& func = method.function;
  const auto& qn = method.exportName;
  if (exportSet.contains(qn)) {
    if (toplevel) {
      exportSet.update(qn, toplevel);
    }
    return;
  }
  auto graph = func.graph()->copy();

  Inline(*graph);

  std::shared_ptr<MobileCode> code;
  code = std::make_shared<MobileCode>(
      graph, func.name(), BytecodeEmitMode::is_default_value_for_unspecified_arg_enabled() /* emit_default_input_instructions */, BytecodeEmitMode::is_default_args_before_out_args_enabled() /* enable_defaults_args_with_out_args */);
  auto instructions_copy = code->instructions();

  exportSet.add(
      qn, ExportedBytecode{std::move(code), func.getSchema(), toplevel});

  auto exports =
      getModuleInterfaceExports(method.module, getInterfaceCalls(*graph));
  for (const auto& method : exports) {
    exportFunction(exportSet, method);
  }
}

void setstateTuple(
    BytecodeExportSet& exportSet,
    const Module& module,
    const IValue& ivalue,
    TypeNameUniquer& type_name_uniquer_,
    bool toplevel = false) {
  if (!ivalue.isObject())
    return;
  auto obj = ivalue.toObject();
  auto type = obj->type();
  if (checkHasValidSetGetState(type)) {
    Function& setstate = type->getMethod("__setstate__");
    auto qn = type_name_uniquer_.getUniqueName(obj->type()).qualifiedName() +
        "." + setstate.name();
    if (exportSet.contains(qn)) {
      return;
    }
    if (auto f = tryToGraphFunction(setstate)) {
      exportFunction(exportSet, ModuleMethod{module, *f, std::move(qn)}, toplevel);
    }
  } else {
    for (size_t i = 0, n = type->numAttributes(); i < n; ++i) {
      setstateTuple(exportSet, module, obj->getSlot(i), type_name_uniquer_);
    }
  }
}

bool isLoweredModule(const Module& m) {
  c10::QualifiedName type_name;
  if (m.type()->name()) {
    type_name = m.type()->name().value();
  }
  bool isLoweredModule = false;
  for (const auto& atom : type_name.atoms()) {
    if (atom == "LoweredModule") {
      isLoweredModule = true;
      break;
    }
  }
  return isLoweredModule;
}

// Check if the global static map of backend debug info
// contains debug info for this module and any of its children.
// If so combine all the maps together and return one.
void getBackendDebugInfoMap(
    const Module& m,
    BackendDebugInfoMapType& debug_map) {
  if (isLoweredModule(m)) {
    auto backend_debug_info =
        m.attr("__backend_debug_info").toCustomClass<PyTorchBackendDebugInfo>();
    const auto& map = backend_debug_info->getDebugInfoMap();
    if (map) {
      debug_map.insert(map.value().begin(), map.value().end());
    }
  }
  for (const auto& c : m.children()) {
    getBackendDebugInfoMap(c, debug_map);
  }
}

SourceRangeRecords getBackendSourceRanges(const Module& m) {
  SourceRangeRecords sr_records;
  if (isLoweredModule(m)) {
    constexpr size_t kSourceRange = 1;
    auto backend_debug_info =
        m.attr("__backend_debug_info").toCustomClass<PyTorchBackendDebugInfo>();
    const auto& map = backend_debug_info->getDebugInfoMap();
    if (map) {
      const auto& map_val = map.value();
      // This map is map of debug handle-to-DebugInfoTuple
      // DebugInfoTuple= <source range, op name, inlined_cs_ptr>
      for (const auto& it : map_val) {
        auto& source_range =
            std::get<kDebugInfoTupleSourceRangeIndex>(it.second);
        sr_records.emplace_back(
            std::numeric_limits<size_t>::max(), source_range);
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        auto cs_ptr = std::get<kDebugInfoTupleInlinedCSIndex>(it.second);
        if (cs_ptr) {
          for (const auto& e : cs_ptr->vec()) {
            // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
            const auto sr = std::get<kSourceRange>(e);
            sr_records.emplace_back(std::numeric_limits<size_t>::max(), sr);
          }
        }
      }
    }
  }
  for (const auto& c : m.children()) {
    const auto& child_sr_records = getBackendSourceRanges(c);
    sr_records.reserve(sr_records.size() + child_sr_records.size());
    std::move(
        child_sr_records.begin(),
        child_sr_records.end(),
        std::back_inserter(sr_records));
  }
  return sr_records;
}

} // namespace

BytecodeExportSet moduleMethodsTuple(
    const Module& module,
    TypeNameUniquer& type_name_uniquer_) {
  BytecodeExportSet exportSet;
  auto methods = module.get_methods();
  // top level methods
  for (const auto& method : methods) {
    const auto& f = toGraphFunction(method.function());
    exportFunction(
        exportSet, ModuleMethod{module, f, f.qualname()}, /* toplevel */ true);
  }

  // __setstate__ of all components
  setstateTuple(exportSet, module, module._ivalue(), type_name_uniquer_, true);

  return exportSet;
}

void SetExportModuleExtraFilesHook(ExportModuleExtraFilesHook hook) {
  GetExtraFilesHook() = std::move(hook);
}

void ScriptModuleSerializer::serialize(
    const Module& module,
    const ExtraFilesMap& extra_files,
    bool bytecode_format,
    bool save_mobile_debug_info) {
  C10_LOG_API_USAGE_ONCE("torch.script.save");
  writeExtraFiles(module, extra_files);
  // Serialize the model object
  writeArchive(
      module._ivalue(),
      /*archive_name=*/"data",
      /*archive_dir=*/"",
      /*tensor_dir=*/"data/");
  // Then we serialize all code info.
  convertTypes(module.type());
  writeFiles("code/");
  // The tensor constants from the code are written to a separate archive
  // so loading the code does not depend on loading the data
  std::vector<IValue> ivalue_constants(
      constant_table_.begin(), constant_table_.end());
  if (bytecode_format) {
    writeArchive(
        c10::ivalue::Tuple::create(ivalue_constants),
        /*archive_name=*/"constants",
        /*archive_dir=*/"",
        /*tensor_dir=*/"constants/",
        /*use_storage_context=*/true);

    writeByteCode(module, save_mobile_debug_info);
  } else {
    writeArchive(
        c10::ivalue::Tuple::create(ivalue_constants),
        /*archive_name=*/"constants",
        /*archive_dir=*/"",
        /*tensor_dir=*/"constants/");
  }
  // Acquires and sets minimum (dynamic) version
  for (auto& item : file_streams_) {
    writer_.setMinVersion(item.value().minVersion());
  }
}

void ScriptModuleSerializer::writeArchive(
    const IValue& value,
    const std::string& archive_name,
    const std::string& archive_dir,
    const std::string& tensor_dir,
    bool use_storage_context) {
  std::vector<char> data;
  // Vector to capture the run-time class types during pickling the IValues
  std::vector<c10::ClassTypePtr> memoizedClassTypes;
  std::vector<std::string> tensor_names;
  // tensors that are already serialized in use_storage_context
  std::unordered_set<std::string> serialized_tensors;
  Pickler data_pickle(
      [&](const char* buf, size_t size) {
        data.insert(data.end(), buf, buf + size);
      },
      nullptr,
      [&](const c10::ClassTypePtr& t) {
        return type_name_uniquer_.getUniqueName(t);
      },
      &memoizedClassTypes,
      [&](const at::Tensor& tensor) {
        // returns a string to use in picker.cpp as storage obj key
        if (use_storage_context) {
          bool already_serialized =
              storage_context_.hasStorage(tensor.storage());
          std::string tensor_name =
              std::to_string(
                  storage_context_.getOrAddStorage(tensor.storage())) +
              ".storage";
          if (already_serialized) {
            // this case is hit when storage has been serialized already
            // from a torch.package context
            serialized_tensors.insert(tensor_name);
          }
          tensor_names.push_back(tensor_name);
        } else {
          tensor_names.push_back(std::to_string(tensor_names.size()));
        }
        return tensor_names.back();
      });
  data_pickle.protocol();
  data_pickle.pushIValue(value);
  data_pickle.stop();
  // write out tensor data
  size_t i = 0;
  std::string prefix = archive_name + "/";

  TORCH_INTERNAL_ASSERT(tensor_names.size() == data_pickle.tensorData().size());

  for (const auto& td : data_pickle.tensorData()) {
    WriteableTensorData writable_td = getWriteableTensorData(td);
    std::string tensor_name = tensor_names[i++];
    if (use_storage_context && serialized_tensors.count(tensor_name)) {
      // storage has been serialzed already, skip
      continue;
    }
    writer_.writeRecord(
        tensor_dir + tensor_name,
        writable_td.data(),
        writable_td.sizeInBytes());
  }

  std::string fname = archive_dir + archive_name + ".pkl";
  writer_.writeRecord(fname, data.data(), data.size());

  // serialize all the captured run-time class types
  for (const c10::ClassTypePtr& wroteType : memoizedClassTypes) {
    convertNamedType(wroteType);
  }
}

void ScriptModuleSerializer::writeExtraFiles(
    const Module& module,
    const ExtraFilesMap& extra_files) {
  // Write out extra files.
  for (const auto& kv : extra_files) {
    const std::string key = "extra/" + kv.first;
    writer_.writeRecord(key, kv.second.data(), kv.second.size());
  }
  auto hook = GetExtraFilesHook();
  if (hook) {
    ExtraFilesMap hook_files = hook(module);
    for (const auto& kv : hook_files) {
      // Checks if the hooked file is already written in extra files,
      //   if so, skips it and warns
      if (extra_files.find(kv.first) != extra_files.end()) {
        TORCH_WARN_ONCE(
            "An extra files hook attempted to write ",
            kv.first,
            " but ",
            "this is already written in extra files and so will be skipped. ",
            "This warning will only appear once per process.");
        continue;
      }
      const std::string key = "extra/" + kv.first;
      writer_.writeRecord(key, kv.second.data(), kv.second.size());
    }
  }
}

void ScriptModuleSerializer::updateSourceRangeTags(
    const SourceRangeRecords& ranges) {
  for (const auto& range : ranges) {
    if (source_range_tags_.find(range.range) == source_range_tags_.end()) {
      source_range_tags_[range.range] = current_source_range_tag_;
      current_source_range_tag_++;
    }
  }
}

void ScriptModuleSerializer::convertTypes(const at::NamedTypePtr& root_type) {
  class_deps_.add(root_type);
  for (size_t i = 0; i < class_deps_.size(); ++i) {
    // note: convertNameType may extend class_deps_, so re-checking .size() is
    // necessary
    convertNamedType(class_deps_[i]);
  }
}

void ScriptModuleSerializer::writeFiles(const std::string& code_dir) {
  current_source_range_tag_ = 0;
  // Mapping of filename => src. We need this because multiple classes may go
  // in the same file (e.g. foo.bar.Baz and foo.bar.Qux)
  for (auto& item : file_streams_) {
    const std::string filename = qualifierToArchivePath(item.key(), code_dir);

    std::string src = item.value().str();

    // Only compress these records if they're not tiny.
    // The cpu cost of generating zip datastructs and compressing isn't
    // well-spent for very small records.
    static constexpr size_t kMinToCompress = 200;

    writer_.writeRecord(
        filename,
        src.c_str(),
        src.size(),
        src.size() > kMinToCompress /*compress*/);

    // Write out the debug information
    std::string debugFilename = filename + ".debug_pkl";
    SourceRangePickler source_range_pickler;
    updateSourceRangeTags(item.value().ranges());
    auto range_data =
        source_range_pickler.pickle(item.value().ranges(), source_range_tags_);
    writer_.writeRecord(
        debugFilename,
        range_data.data(),
        range_data.size(),
        range_data.size() > kMinToCompress /*compress*/);
  }
}

void ScriptModuleSerializer::writeByteCode(
    const Module& module,
    const bool save_mobile_debug_info) {
  std::vector<c10::IValue> elements;
  BackendDebugInfoRecorder debug_info_recorder;
  int64_t version_to_write = caffe2::serialize::kProducedBytecodeVersion;

  elements.emplace_back(static_cast<int64_t>(version_to_write));
  std::vector<c10::IValue> debug_info_elements;
  // Always save debug handles
  debug_info_elements.emplace_back(static_cast<int64_t>(version_to_write));

  BytecodeExportSet exportSet = moduleMethodsTuple(module, type_name_uniquer_);
  exportSet.exportIValues(
      elements, debug_info_elements, debug_info_recorder, type_name_uniquer_);

  auto telements = to_tuple(std::move(elements));
  writeArchive(
      telements,
      /*archive_name=*/"bytecode",
      /*archive_dir=*/"",
      /*tensor_dir=*/"constants/",
      /*use_storage_context=*/true);

  auto debug_info_telements = to_tuple(std::move(debug_info_elements));

  // At the moment keeping this feature experimental
  // since we have not evaluated how this affect model size
  // and we have not build any utility to strip off debug info
  // when desired
  // TODO: Build utility to strip off debug map. It should also do the
  // same for debug_pkl files
  if (save_mobile_debug_info) {
    // Note that stripping off debug map will not strip off
    // debug handles.
    // The reason we save debug handles conditionally is so that
    // we dont end up with a model that has debug handles but has not
    // debug map to correlate debug handels with.
    // Once we have a model with both handles and debug map, we can
    // strip off debug map and have a lean model served to production.
    // If exception ocurrs we have a model with debug map that can be
    // used to symbolicate debug handles
    writeArchive(
        debug_info_telements,
        /*archive_name=*/"mobile_debug_handles",
        /*archive_dir=*/"",
        /*tensor_dir=*/"mobile_debug_handles/");
    static constexpr size_t kMinToCompress = 200;
    // For delegated backends get source ranges that are in the debug info
    // map. Since delegated backend replace original module with lowered
    // module we will not serialize original module's code which is what would
    // have contained source range. Since we dont have that anymore, extract
    // source ranges out of delegated module and store in a separate archive.
    // Note that we must do this first because in order to serialize inlined
    // CS appropriate source_range_tags must have been generated.
    auto backend_source_range_records = getBackendSourceRanges(module);
    SourceRangePickler source_range_pickler;
    updateSourceRangeTags(backend_source_range_records);
    auto range_data = source_range_pickler.pickle(
        backend_source_range_records, source_range_tags_);
    std::string debugFilename = "delegated_backends.debug_pkl";
    writer_.writeRecord(
        debugFilename,
        range_data.data(),
        range_data.size(),
        range_data.size() > kMinToCompress /*compress*/);

    // For delegated backends get debug_info_map
    // This is merged with other debug_info_map of other modules
    // which were not delegated.
    BackendDebugInfoMapType backend_debug_info_map;
    getBackendDebugInfoMap(module, backend_debug_info_map);
    // Now get the debug-handles-to-inlined-cs-ptr-map
    // And serialize that in a separate archive
    auto debug_handle_cs_ptr_map = debug_info_recorder.stopRecording();
    debug_handle_cs_ptr_map.insert(
        backend_debug_info_map.begin(), backend_debug_info_map.end());
    CallStackDebugInfoPickler cs_debug_info_pickler;
    auto cs_data = cs_debug_info_pickler.pickle(
        debug_handle_cs_ptr_map, source_range_tags_);
    // Write out map: [debug-handle, {source range, InlinedCallStack}]
    std::string filename = "callstack_debug_map.pkl";
    writer_.writeRecord(
        filename,
        cs_data.data(),
        cs_data.size(),
        cs_data.size() > kMinToCompress /*compress*/);
  }
}

void ScriptModuleSerializer::convertNamedType(
    const c10::NamedTypePtr& class_type) {
  if (converted_types_.count(class_type)) {
    return;
  }
  converted_types_.insert(class_type);
  auto qualname = type_name_uniquer_.getUniqueName(class_type);
  std::string qualifier = qualname.prefix();
  PythonPrint* pp = file_streams_.find(qualifier);

  auto type_printer =
      [&](const c10::ConstTypePtr& t) -> c10::optional<std::string> {
    auto namedType = t->cast<c10::NamedType>();
    if (namedType && namedType->name()) {
      return type_name_uniquer_.getUniqueName(namedType).qualifiedName();
    }
    return c10::nullopt;
  };
  if (!pp) {
    pp = &file_streams_.insert(
        std::move(qualifier),
        PythonPrint(
            constant_table_,
            class_deps_,
            type_printer,
            /*enforce_importable=*/true));
  }
  pp->printNamedType(class_type);
}

void ScriptModuleSerializer::serialize_unified_format(
    Module& module,
    uint64_t script_module_id) {
  const std::string archive_dir =
      ".data/ts_code/" + std::to_string(script_module_id) + "/";

  // Serialize the model object
  writeArchive(
      module._ivalue(),
      "data",
      archive_dir,
      /*tensor_dir=*/".data/",
      /*use_storage_context=*/true);
  // Then we serialize all code info.
  convertTypes(module.type());
  // The tensor constants from the code are written to a separate archive
  // so loading the code does not depend on loading the data
  std::vector<IValue> ivalue_constants(
      constant_table_.begin(), constant_table_.end());
  writeArchive(
      c10::ivalue::Tuple::create(ivalue_constants),
      "constants",
      archive_dir,
      /*tensor_dir=*/".data/",
      /*use_storage_context=*/true);

  // Note: writeFiles() call needs to be made in addition to calling this
  // function to have the code actually saved (tensors are saved)
}

SerializationStorageContext& ScriptModuleSerializer::storage_context() {
  return storage_context_;
}

void ExportModule(
    const Module& module,
    std::ostream& out,
    const ExtraFilesMap& extra_files,
    bool bytecode_format,
    bool save_mobile_debug_info) {
  caffe2::serialize::PyTorchStreamWriter writer(
      [&](const void* buf, size_t nbytes) -> size_t {
        out.write(static_cast<const char*>(buf), nbytes);
        return !out ? 0 : nbytes;
      });
  ScriptModuleSerializer serializer(writer);
  serializer.serialize(
      module, extra_files, bytecode_format, save_mobile_debug_info);
}

void ExportModule(
    const Module& module,
    const std::string& filename,
    const ExtraFilesMap& extra_files,
    bool bytecode_format,
    bool save_mobile_debug_info) {
  caffe2::serialize::PyTorchStreamWriter writer(filename);
  ScriptModuleSerializer serializer(writer);
  serializer.serialize(
      module, extra_files, bytecode_format, save_mobile_debug_info);
}

void ExportModule(
    const Module& module,
    const std::function<size_t(const void*, size_t)>& writer_func,
    const ExtraFilesMap& extra_files,
    bool bytecode_format,
    bool save_mobile_debug_info) {
  caffe2::serialize::PyTorchStreamWriter writer(writer_func);
  ScriptModuleSerializer serializer(writer);
  serializer.serialize(
      module, extra_files, bytecode_format, save_mobile_debug_info);
}

namespace {
void export_opnames(const script::Module& m, std::set<std::string>& opnames) {
  std::vector<c10::IValue> elements;
  BackendDebugInfoRecorder dummy;
  TypeNameUniquer dummy_uniquer = TypeNameUniquer();
  BytecodeExportSet exportSet = moduleMethodsTuple(m, dummy_uniquer);
  exportSet.exportIValues(elements, dummy, dummy_uniquer);
  for (const auto& element : elements) {
    auto table = element.toTuple()->elements()[1];
    auto row =
        table.toTuple()->elements().at(BYTECODE_INDEX_OPERATOR).toTuple();
    TORCH_INTERNAL_ASSERT(
        row->elements().at(0).toStringRef() == "operators",
        "Expected operators but found ",
        row->elements().at(0).toStringRef());
    const auto& ops_list = row->elements().at(1).toTuple()->elements();
    for (const auto& op : ops_list) {
      auto op_item = op.toTuple()->elements();
      TORCH_CHECK(
          op_item.size() >= 2,
          "There should be either two parts (name and overload name), ",
          "or three parts (name, overload name and number of specified args) ",
          "for an operator.");
      auto opname = op_item[0].toString()->string();
      auto overload = op_item[1].toString()->string();
      // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
      opnames.emplace(overload.empty() ? opname : opname + "." + overload);
    }
  }
}
} // namespace

std::vector<std::string> export_opnames(const script::Module& m) {
  std::set<std::string> names;
  export_opnames(m, names);
  return std::vector<std::string>(names.begin(), names.end());
}

// Thread local flag (only happens in export, i.e. on server side)
// to control if instructions for bytecode default inputs are emitted
// or not. It's the major difference between bytecode v5 and v6.
thread_local bool emitBytecodeDefaultInputs =
    caffe2::serialize::kProducedBytecodeVersion <= 5 ? true : false;
bool BytecodeEmitMode::is_default_value_for_unspecified_arg_enabled() {
  return emitBytecodeDefaultInputs;
}
void BytecodeEmitMode::set_default_value_for_unspecified_arg_enabled(
    bool enabled) {
  emitBytecodeDefaultInputs = enabled;
}

thread_local bool emitDefautlArgsWithOutArgs =
    caffe2::serialize::kProducedBytecodeVersion <= 6 ? false : true;
bool BytecodeEmitMode::is_default_args_before_out_args_enabled() {
  return emitDefautlArgsWithOutArgs;
}
void BytecodeEmitMode::set_default_args_before_out_args_enabled(bool enabled) {
  emitDefautlArgsWithOutArgs = enabled;
}

} // namespace jit
} // namespace torch
