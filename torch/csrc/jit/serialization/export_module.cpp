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
#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>
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
#include <cerrno>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch::jit {

CompilationOptions getOptionsFromGlobal() {
  CompilationOptions compilation_options;
  compilation_options.enable_default_args_before_out_args =
      BytecodeEmitMode::is_default_args_before_out_args_enabled();
  compilation_options.enable_default_value_for_unspecified_arg =
      BytecodeEmitMode::is_default_value_for_unspecified_arg_enabled();
  compilation_options.enable_emit_promoted_ops =
      BytecodeEmitMode::is_emit_promoted_ops_enabled();
  compilation_options.incl_interface_call = getMobileInterfaceCallExport();
  compilation_options.model_version =
      caffe2::serialize::kProducedBytecodeVersion;
  return compilation_options;
}

static IValue to_tuple(std::initializer_list<IValue> ivalues) {
  return c10::ivalue::Tuple::create(ivalues);
}

IValue to_tuple(std::vector<IValue> ivalues) {
  return c10::ivalue::Tuple::create(std::move(ivalues));
}

IValue Table(const std::vector<std::pair<std::string, IValue>>& entries) {
  std::vector<IValue> ivalue_entries;
  ivalue_entries.reserve(entries.size());
  for (const auto& e : entries) {
    ivalue_entries.push_back(to_tuple({e.first, e.second}));
  }
  return to_tuple(std::move(ivalue_entries));
}

namespace {

ExportModuleExtraFilesHook& GetExtraFilesHook() {
  static ExportModuleExtraFilesHook func = nullptr;
  return func;
}

/**
 * If the type is not NamedTuple, it will return default_type_str. If the type
 * is a NamedTuple, it will return a string with following structure to describe
 * the content in the NamedTuple: "qualified_named[ NamedTuple, [ [filed_name_1,
 * field_type_1], [filed_name_2, field_type_2]
 *   ]
 * ]"
 *  Example NamedTuple type:
 *  "__torch__.base_models.sparse_nn.pytorch_preproc_types.PreprocOutputType[
 *     NamedTuple, [
 *         [float_features, Tensor],
 *         [id_list_features, List[Tensor]],
 *         [label,  Tensor],
 *         [weight, Tensor],
 *         ]
 *     ]"
 *
 * @param compilation_unit Jit compilation unit to look up function schema.
 * @param type_ptr A type pointer and it can be possibly any type.
 * @param default_type_str The default string representation. The string can
 * either from type_ptr->str(), type_ptr->annotation_str(), or
 * type_ptr->repr_str(). In some cases, they could be different in different
 * scenario. For example, Tensor type can be "Tensor", "Tensor (inferred)" and
 * "Tensor[]", and we only want "Tensor". Leave it as part of arguments as the
 * default return, when type_ptr is not a NamedTuple.
 * @return string representation.
 */
std::string get_named_tuple_str_or_default(
    const CompilationUnit& compilation_unit,
    const TypePtr& type_ptr,
    std::string default_type_str) {
  if (type_ptr->kind() == TypeKind::TupleType) {
    // For the simple types (Tensor, Tensor), the mobile type parse can parse
    // it and compilation unit won't have it's definition. The default type
    // string will be returned instead.
    if (compilation_unit.get_named_tuple(type_ptr->str())) {
      auto named_tuple_ptr = compilation_unit.get_named_tuple(type_ptr->str());
      if (named_tuple_ptr != nullptr) {
        std::string named_tuple_str = type_ptr->str();
        named_tuple_str.append("[NamedTuple, [");
        std::vector<IValue> name_type_pairs;

        // Get the field name and field type for the NamedTuple
        for (auto it = named_tuple_ptr->schema()->arguments().begin();
             it != named_tuple_ptr->schema()->arguments().end();
             it++) {
          const std::string named_tuple_name = it->name();
          const c10::TypePtr& named_tuple_type = it->type();
          // When it->type() is Tensor type, in Python, if it's inferred type,
          // str() return "Tensor" and repr_str() return "Tensor (inferred)". If
          // it's not inferred type, str() return "Tensor[]" and repr_str()
          // return "Tensor". In cpp, repr_str() will always return "Tensor"
          // regardless inferred type. When exporing custom type in bytecode,
          // "Tensor" is the preferred way to deserialize Tensor type
          std::string named_tuple_type_str = it->is_inferred_type()
              ? named_tuple_type->str()
              : named_tuple_type->repr_str();
          // The type can also be NamedTuple. Will parse it recursively and get
          // it's string representation.
          named_tuple_type_str = get_named_tuple_str_or_default(
              compilation_unit, named_tuple_type, named_tuple_type_str);
          name_type_pairs.emplace_back(
              c10::ivalue::Tuple::create({it->name(), named_tuple_type_str}));

          named_tuple_str.append("[")
              .append(named_tuple_name)
              .append(", ")
              .append(named_tuple_type_str)
              .append("]");
          if (it != named_tuple_ptr->schema()->arguments().end() - 1) {
            named_tuple_str.append(",");
          }
        }
        named_tuple_str.append("]]");
        return named_tuple_str;
      }
    }
  }
  return default_type_str;
}

std::pair<IValue, IValue> getFunctionTuple(
    const CompilationUnit& compilation_unit,
    const mobile::Function& func,
    BackendDebugInfoRecorder& debug_info_recorder,
    TypeNameUniquer& type_name_uniquer_) {
  const auto& mobile_code = func.get_code();

  // instructions
  std::vector<IValue> instructions;
  instructions.reserve(mobile_code.instructions_.size());
  for (Instruction ins : mobile_code.instructions_) {
    instructions.emplace_back(to_tuple({toString(ins.op), ins.X, ins.N}));
  }

  // operators
  std::vector<IValue> operators;
  operators.reserve(mobile_code.op_names_.size());
  for (const auto i : c10::irange(mobile_code.op_names_.size())) {
    const auto& opname = mobile_code.op_names_[i];
    const int size = mobile_code.operator_input_sizes_[i];
    if (BytecodeEmitMode::is_default_value_for_unspecified_arg_enabled()) {
      operators.emplace_back(to_tuple({opname.name, opname.overload_name}));
    } else {
      operators.emplace_back(
          to_tuple({opname.name, opname.overload_name, size}));
    }
  }

  // types
  std::vector<IValue> types;
  types.reserve(mobile_code.types_.size());
  static const std::string torch_prefix("__torch__");
  static const std::string class_prefix("__torch__.torch.classes");

  for (const TypePtr& ty : mobile_code.types_) {
    auto t = ty;
    if (auto dyn = t->castRaw<c10::DynamicType>()) {
      t = dyn->fallback();
    }
    std::string type_str = t->annotation_str();
    if (t->kind() == TypeKind::DictType) {
      // For DictType, there are two items in t->containedTypes(), the first one
      // is key and the second one is value. Both of them could be NamedTuple
      // type.
      const TypePtr& key_type = t->containedTypes()[0];
      const TypePtr& value_type = t->containedTypes()[1];
      std::string key_type_str = get_named_tuple_str_or_default(
          compilation_unit, key_type, key_type->annotation_str());
      std::string value_type_str = get_named_tuple_str_or_default(
          compilation_unit, value_type, value_type->annotation_str());

      // Construct the dict representation after achieving correct string
      // representation for both key and value, like
      // "Dict[str,__torch__.dper3.core.pytorch_schema_utils.IdScoreListFeatureTuple[NamedTuple,
      // [[lengths, Tensor],[values,
      // __torch__.dper3.core.pytorch_schema_utils.IdScoreTuple[NamedTuple,
      // [[ids, Tensor],[scores, Tensor]]]],[offsets, Optional[Tensor]]]]]"
      std::string dict_str;
      dict_str.append("Dict[")
          .append(key_type_str)
          .append(",")
          .append(value_type_str)
          .append("]");
      types.emplace_back(dict_str);
      continue;
    } else if (t->kind() == TypeKind::TupleType) {
      std::string named_tuple_str =
          get_named_tuple_str_or_default(compilation_unit, t, type_str);
      types.emplace_back(named_tuple_str);
      continue;
    } else if (type_str.find(torch_prefix) == 0) {
      TORCH_CHECK(
          type_str.find(class_prefix) == 0,
          "__torch__ types other than custom c++ classes (__torch__.torch.classes)"
          "are not supported in lite interpreter. ",
          "Workaround: instead of using arbitrary class type (class Foo()), ",
          "define a pytorch class (class Foo(torch.nn.Module)). The problematic type is: ",
          type_str);
    }
    types.emplace_back(type_str);
  }

  // since the register location is embedded into the bytecode, pass the
  // register size
  auto register_size = static_cast<int>(mobile_code.register_size_);

  auto codeTable = Table(
      {{"instructions", to_tuple(instructions)},
       {"operators", to_tuple(operators)},
       {"constants", to_tuple(mobile_code.constants_)},
       {"types", to_tuple(types)},
       {"register_size", register_size}});

  // schema
  const auto& schema = func.getSchema();
  auto type_printer = [&](const c10::Type& t) -> c10::optional<std::string> {
    auto namedType = t.cast<c10::NamedType>();
    if (namedType && namedType->name()) {
      return type_name_uniquer_.getUniqueName(namedType).qualifiedName();
    }
    return c10::nullopt;
  };

  auto makeArgTuple = [&](const std::vector<Argument>& args) {
    std::vector<IValue> argTables;
    for (auto&& arg : args) {
      TORCH_CHECK(
          !arg.N(),
          "Arguments with known list lengths are not supported in mobile modules.");
      TORCH_CHECK(
          !arg.kwarg_only(),
          "Keyword-only arguments are not supported in mobile modules.");
      /*
        This part adds the argument's name, type and default_value in
        `bytecode.pkl` This has to be consistent with the `code/` directory
        which has annotated py code of the entire module. `type_printer` uses
        `TypeNameUniquer` to get the managled name of the argument. This helps
        in having the right object reference when a class method is called using
        the `self` argument.

        arg.type()->annotation_str(type_printer) => mangled unique name of the
        module/submodule
      */
      auto arg_type = arg.type();
      if (auto dyn = arg_type->castRaw<c10::DynamicType>()) {
        arg_type = dyn->fallback();
      }
      argTables.emplace_back(Table({
          {"name", arg.name()},
          {"type", arg_type->annotation_str(type_printer)},
          {"default_value", arg.default_value()},
      }));
    }
    return to_tuple(argTables);
  };
  auto schemaTable = Table({
      {"arguments", makeArgTuple(schema.arguments())},
      {"returns", makeArgTuple(schema.returns())},
  });

  // function tuple
  std::string qn;
  if (func.name() == "__setstate__" || func.name() == "__getstate__") {
    auto classtype = func.getSchema().arguments()[0].type()->cast<ClassType>();
    TORCH_INTERNAL_ASSERT(
        classtype, "class is null ", func.qualname().qualifiedName());
    qn = c10::QualifiedName(
             type_name_uniquer_.getUniqueName(classtype), func.name())
             .qualifiedName();
  } else {
    qn = func.qualname().qualifiedName();
  }
  auto bytecode_vals = to_tuple({qn, codeTable, schemaTable});

  c10::optional<IValue> debug_info_vals;
  // module debug info
  // This is just a set of debug handles.
  // We always save debug handles.
  // debug handles generated by debug_handle_manager
  // will correspond to {source_range, inlinedCallStackPtr} which we will
  // serialize separately.
  IValue module_debug_tuple =
      c10::ivalue::Tuple::create(mobile_code.debug_handles_);
  auto function_debug_info =
      Table({{"function_debug_handles", module_debug_tuple}});
  debug_info_vals = to_tuple({qn, function_debug_info});
  return std::make_pair(bytecode_vals, debug_info_vals);
}

void pushMobileFunctionsToIValues(
    const CompilationUnit& compilation_unit,
    const mobile::Module& module,
    std::vector<c10::IValue>& elements,
    std::vector<c10::IValue>& debugInfoElements,
    BackendDebugInfoRecorder& recorder,
    TypeNameUniquer& uniquer) {
  for (const auto& method : module.get_methods()) {
    auto tuple = getFunctionTuple(
        compilation_unit, method.function(), recorder, uniquer);
    elements.push_back(std::move(tuple.first));
    debugInfoElements.push_back(std::move(tuple.second));
  }
}

struct ModuleMethod {
  ModuleMethod(const Module& m, const GraphFunction& f, c10::QualifiedName n)
      : module(m), function(f), exportName(std::move(n)) {}
  Module module;
  const GraphFunction& function;
  c10::QualifiedName exportName;
};

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

// TODO: remove mobileInterfaceCallExport as it is no longer needed.
// This function was introduced to guard the usage of `InterfaceCall` and
// now the support for `InterfaceCall` should be mature enough.
auto& mobileInterfaceCallExport() {
  static std::atomic<bool> flag{true};
  return flag;
}

} // namespace

TORCH_API void enableMobileInterfaceCallExport() {
  mobileInterfaceCallExport().store(true, std::memory_order_relaxed);
}
bool getMobileInterfaceCallExport() {
  return mobileInterfaceCallExport().load(std::memory_order_relaxed);
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
  if (module.retrieve_traced_inputs().size() > 0) {
    writeArchive(
        module.retrieve_traced_inputs(),
        /*archive_name=*/"traced_inputs",
        /*archive_dir=*/"",
        /*tensor_dir=*/"traced_inputs/",
        /*use_storage_context*/ false,
        /*skip_tensor_data*/ true);
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
    bool use_storage_context,
    bool skip_tensor_data) {
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
    std::string tensor_name = tensor_names[i++];
    if (td.is_meta() || skip_tensor_data) {
      writer_.writeRecord(tensor_dir + tensor_name, nullptr, 0);
      continue;
    }
    WriteableTensorData writable_td = getWriteableTensorData(td);
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

  mobile::Module mobile_module =
      jitModuleToMobile(module, getOptionsFromGlobal());

  pushMobileFunctionsToIValues(
      *module._ivalue()->compilation_unit(),
      mobile_module,
      elements,
      debug_info_elements,
      debug_info_recorder,
      type_name_uniquer_);

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
    const auto& debug_info = mobile_module.getDebugTable().getCallStackPtrMap();
    BackendDebugInfoMapType debug_handle_cs_ptr_map(
        debug_info.begin(), debug_info.end());
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

namespace {

c10::optional<std::string> type_printer(
    const c10::Type& type,
    torch::jit::TypeNameUniquer& type_name_uniquer) {
  if (auto dyn = type.castRaw<c10::DynamicType>()) {
    return dyn->fallback()->annotation_str(
        [&](auto&& t) { return type_printer(t, type_name_uniquer); });
  }
  auto namedType = type.cast<c10::NamedType>();
  if (namedType && namedType->name()) {
    return type_name_uniquer.getUniqueName(namedType).qualifiedName();
  }
  return c10::nullopt;
}

} // namespace

void ScriptModuleSerializer::convertNamedType(
    const c10::NamedTypePtr& class_type) {
  if (converted_types_.count(class_type)) {
    return;
  }
  converted_types_.insert(class_type);
  auto qualname = type_name_uniquer_.getUniqueName(class_type);
  std::string qualifier = qualname.prefix();
  PythonPrint* pp = file_streams_.find(qualifier);

  if (!pp) {
    pp = &file_streams_.insert(
        std::move(qualifier),
        PythonPrint(
            constant_table_,
            class_deps_,
            [&](const c10::Type& t) {
              return type_printer(t, type_name_uniquer_);
            },
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
    bool save_mobile_debug_info,
    bool use_flatbuffer) {
  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
    out.write(static_cast<const char*>(buf), nbytes);
    return !out ? 0 : nbytes;
  };
  ExportModule(
      module,
      writer_func,
      extra_files,
      bytecode_format,
      save_mobile_debug_info,
      use_flatbuffer);
}

void ExportModule(
    const Module& module,
    const std::string& filename,
    const ExtraFilesMap& extra_files,
    bool bytecode_format,
    bool save_mobile_debug_info,
    bool use_flatbuffer) {
  if (!use_flatbuffer) {
    // the zip archive need to know the filepath
    caffe2::serialize::PyTorchStreamWriter writer(filename);
    ScriptModuleSerializer serializer(writer);
    serializer.serialize(
        module, extra_files, bytecode_format, save_mobile_debug_info);
    return;
  }
  std::ofstream ofile;
  ofile.open(filename, std::ios::binary | std::ios::out);
  if (ofile.fail()) {
    std::stringstream message;
    if (errno == ENOENT) {
      message << "Parent directory of " << filename << " does not exist.\n";
    } else {
      message << "Error while opening file: " << errno << std::endl;
      ;
    }
    TORCH_CHECK(false, message.str());
  }
  ExportModule(
      module,
      ofile,
      extra_files,
      bytecode_format,
      save_mobile_debug_info,
      use_flatbuffer);
}

void save_jit_module(
    const Module& module,
    const std::string& filename,
    const ExtraFilesMap& extra_files) {
  auto buffer = save_jit_module_to_bytes(module, extra_files);
  std::fstream ofile(filename, std::ios::binary | std::ios::out);
  ofile.write(
      reinterpret_cast<char*>(buffer->data()), buffer->size()); // NOLINT
  ofile.close();
}

DetachedBuffer::UniqueDetachedBuffer save_jit_module_to_bytes(
    const Module& module,
    const ExtraFilesMap& extra_files) {
  ExtraFilesMap jitfiles;
  std::vector<IValue> constants;
  jitModuleToPythonCodeAndConstants(module, &jitfiles, &constants);
  CompilationOptions options = getOptionsFromGlobal();
  mobile::Module mobilem = jitModuleToMobile(module, options);
  return save_mobile_module_to_bytes(mobilem, extra_files, jitfiles, constants);
}

void save_jit_module_to_write_func(
    const Module& module,
    const ExtraFilesMap& extra_files,
    bool save_mobile_debug_info,
    const std::function<size_t(const void*, size_t)>& writer_func) {
  (void)save_mobile_debug_info;
  auto buffer = save_jit_module_to_bytes(module, extra_files);
  writer_func(reinterpret_cast<void*>(buffer->data()), buffer->size());
}

void ExportModule(
    const Module& module,
    const std::function<size_t(const void*, size_t)>& writer_func,
    const ExtraFilesMap& extra_files,
    bool bytecode_format,
    bool save_mobile_debug_info,
    bool use_flatbuffer) {
  if (use_flatbuffer) {
    save_jit_module_to_write_func(
        module, extra_files, save_mobile_debug_info, writer_func);
  } else {
    caffe2::serialize::PyTorchStreamWriter writer(writer_func);
    ScriptModuleSerializer serializer(writer);
    serializer.serialize(
        module, extra_files, bytecode_format, save_mobile_debug_info);
  }
}

namespace {
void export_opnames(const script::Module& m, std::set<std::string>& opnames) {
  mobile::Module mobile_m = jitModuleToMobile(m, getOptionsFromGlobal());
  for (const auto& method : mobile_m.get_methods()) {
    for (const auto& op : method.function().get_code().op_names_) {
      // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
      opnames.emplace(
          op.overload_name.empty() ? op.name
                                   : op.name + "." + op.overload_name);
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

thread_local bool emitDefaultEmitPromotedOps =
    caffe2::serialize::kProducedBytecodeVersion <= 7 ? false : true;
bool BytecodeEmitMode::is_emit_promoted_ops_enabled() {
  return emitDefaultEmitPromotedOps;
}
void BytecodeEmitMode::set_default_emit_promoted_ops_enabled(bool enabled) {
  emitDefaultEmitPromotedOps = enabled;
}

} // namespace torch::jit
