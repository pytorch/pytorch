#include <torch/csrc/jit/serialization/export.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/backends/backend_debug_handler.h>
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
#include <torch/csrc/jit/serialization/import_export_constants.h>
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
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {

static constexpr const char* kArchiveNameConstants = "constants";
static constexpr const char* kArchiveNameBytecode = "bytecode";

char const* toString(OpCode op);

namespace {

ExportModuleExtraFilesHook& GetExtraFilesHook() {
  static ExportModuleExtraFilesHook func = nullptr;
  return func;
}

ExportModuleMobileInfoConverter& GetMobileInfoConverter() {
  static ExportModuleMobileInfoConverter func = nullptr;
  return func;
}

static IValue Tup(std::vector<IValue> ivalues) {
  return c10::ivalue::Tuple::create(std::move(ivalues));
}

static IValue Table(
    const std::vector<std::pair<std::string, IValue>>& entries) {
  std::vector<IValue> ivalue_entries;
  ivalue_entries.reserve(entries.size());
  for (const auto& e : entries) {
    ivalue_entries.push_back(Tup({e.first, e.second}));
  }
  return Tup(std::move(ivalue_entries));
}

std::pair<IValue, IValue> getFunctionTuple(
    const Module& module,
    const Function& func,
    BackendDebugHandleManager& debug_handle_manager) {
  auto graph = func.graph()->copy();

  Inline(*graph);

  torch::jit::MobileCode code(graph, func.name());
  auto instructions_copy = code.instructions();

  // operator names
  std::vector<c10::OperatorName> opnames;
  std::vector<std::string> method_names;
  std::vector<int64_t> op_debug_handles;
  for (size_t i = 0; i < instructions_copy.size(); ++i) {
    Instruction ins = instructions_copy[i];
    if (ins.op == OP || ins.op == OPN) {
      auto node = code.instructions_source()[i];
      opnames.emplace_back(node->schema().operator_name());
    }
    // CALL nodes at this point represent built-in (i.e. non-Graph)
    // functions that were not inlined. Here we convert the CALL
    // instructions for these functions into INTERFACE_CALL instructions
    // s.t. at runtime, we will look up the Function* on the Type of the
    // 0th argument in the stack and call that directly.
    if (ins.op == CALL) {
      auto node = code.instructions_source()[i];
      if (node->kind() == prim::CallMethod) {
        // NB: replacing instruction
        auto method_name_idx =
            code.constant_table().size() + method_names.size();
        method_names.emplace_back(node->s(attr::name));
        Instruction new_instr{
            INTERFACE_CALL,
            static_cast<int32_t>(method_name_idx),
            static_cast<uint16_t>(node->inputs().size())};
        instructions_copy[i] = new_instr;
      } else {
        TORCH_INTERNAL_ASSERT(
            false, "Unsupported node kind on CALL opcode for mobile");
      }
    } else if (ins.op == RET) {
      auto node = code.instructions_source()[i];
      for (const auto& input : node->inputs()) {
        const auto& input_type = input->type();
        if (input_type->kind() == TypeKind::TupleType) {
          if (const auto& name_typed_input =
                  input_type->cast<at::NamedType>()) {
            TORCH_CHECK(
                !name_typed_input->name(),
                "A named tuple type is not supported in mobile module. ",
                "Workaround: instead of using a named tuple type's fields, ",
                "use a dictionary type's key-value pair itmes or ",
                "a pytorch class (class Foo(torch.nn.Module))'s attributes.'");
          }
        } else if (
            input_type->kind() == TypeKind::ListType ||
            input_type->kind() == TypeKind::DictType) {
          for (const TypePtr& element_type : input_type->containedTypes()) {
            TORCH_CHECK(
                element_type->kind() != TypeKind::ClassType,
                "Returining a list or dictionary with pytorch class type ",
                "is not supported in mobile module "
                "(List[Foo] or Dict[int, Foo] for class Foo(torch.nn.Module)). "
                "Workaround: instead of using pytorch class as their element type, ",
                "use a combination of list, dictionary, and single types.");
          }
        }
      }
    } else {
      TORCH_CHECK(
          isOpSupportedInMobile(ins.op),
          toString(ins.op),
          " is not supported in mobile module.");
    }
    auto node = code.instructions_source()[i];
    int64_t debug_handle =
        debug_handle_manager.getNextDebugHandleForInlinedCallStackPtr(node);
    // Note 1-to-1 correspondence between instructions and debug handles
    op_debug_handles.emplace_back(debug_handle);
  }

  // instructions
  std::vector<IValue> instructions;
  instructions.reserve(instructions_copy.size());
  for (Instruction ins : instructions_copy) {
    instructions.emplace_back(Tup({toString(ins.op), ins.X, ins.N}));
  }

  // operators
  std::vector<IValue> operators;
  operators.reserve(opnames.size());
  for (const auto& opname : opnames) {
    operators.emplace_back(Tup({opname.name, opname.overload_name}));
  }

  // constants
  //
  // Make a copy of the constants and append the method names
  // that we emitted for the converted INTERFACE_CALL nodes above.
  auto constants = code.constant_table();
  for (auto& method_name : method_names) {
    constants.emplace_back(std::move(method_name));
  }

  // types
  std::vector<IValue> types;
  types.reserve(code.type_table().size());
  static const std::string torch_prefix("__torch__");
  static const std::string class_prefix("__torch__.torch.classes");
  for (const TypePtr& t : code.type_table()) {
    auto type_str = t->annotation_str();
    if (type_str.find(torch_prefix) == 0) {
      TORCH_CHECK(
          type_str.find(class_prefix) == 0,
          "__torch__ types other than torchbind (__torch__.torch.classes)"
          "are not supported in lite interpreter. ",
          "Workaround: instead of using arbitrary class type (class Foo()), ",
          "define a pytorch class (class Foo(torch.nn.Module)).");
    }
    types.emplace_back(type_str);
  }

  // since the register location is embedded into the bytecode, pass the
  // register size
  auto register_size = static_cast<int>(code.register_size());

  auto codeTable = Table(
      {{"instructions", Tup(instructions)},
       {"operators", Tup(operators)},
       {kArchiveNameConstants, Tup(constants)},
       {"types", Tup(types)},
       {"register_size", register_size}});

  // schema
  const auto& schema = func.getSchema();
  TORCH_CHECK(
      schema.overload_name().empty(), // @TODO: is this check correct?
      "Overloads are not supported in mobile modules.");
  TORCH_CHECK(
      !schema.is_vararg(), "Python *args are not supported in mobile modules.");
  TORCH_CHECK(
      !schema.is_varret(),
      "A variable number of return values is not supported in mobile modules.");
  auto makeArgTuple = [](const std::vector<Argument>& args) {
    std::vector<IValue> argTables;
    for (auto&& arg : args) {
      TORCH_CHECK(
          !arg.N(),
          "Arguments with known list lengths are not supported in mobile modules.");
      TORCH_CHECK(
          !arg.kwarg_only(),
          "Keyword-only arguments are not supported in mobile modules.");
      argTables.emplace_back(Table({
          {"name", arg.name()},
          {"type", arg.type()->annotation_str()},
          {"default_value", arg.default_value()},
      }));
    }
    return Tup(argTables);
  };
  auto schemaTable = Table({
      {"arguments", makeArgTuple(schema.arguments())},
      {"returns", makeArgTuple(schema.returns())},
  });

  // function tuple
  auto bytecode_vals =
      Tup({func.qualname().qualifiedName(), codeTable, schemaTable});

  c10::optional<IValue> debug_info_vals;
  // module debug info
  // This is just a set of debug handles.
  // We always save debug handles.
  // debug handles generated by debug_handle_manager
  // will correspond to {source_range, inlinedCallStackPtr} which we will
  // serialize separately.
  IValue module_debug_tuple = c10::ivalue::Tuple::create(op_debug_handles);
  auto function_debug_info =
      Table({{"function_debug_handles", module_debug_tuple}});
  debug_info_vals = Tup({func.qualname().qualifiedName(), function_debug_info});
  return std::make_pair(bytecode_vals, debug_info_vals);
}

void setstateTuple(
    const Module& module,
    const IValue& ivalue,
    std::vector<c10::IValue>& elements,
    std::unordered_set<std::string>& qn_cache,
    std::vector<c10::IValue>& debug_info_elements,
    BackendDebugHandleManager& debug_handle_manager) {
  if (!ivalue.isObject())
    return;
  auto obj = ivalue.toObject();
  auto type = obj->type();
  if (checkHasValidSetGetState(type)) {
    Function& setstate = type->getMethod("__setstate__");
    const auto qn = setstate.qualname().qualifiedName();
    if (qn_cache.find(qn) != qn_cache.end()) {
      return;
    }
    if (setstate.isGraphFunction()) {
      auto func_tuple =
          getFunctionTuple(module, setstate, debug_handle_manager);
      elements.push_back(func_tuple.first);
      qn_cache.emplace(qn);
      debug_info_elements.push_back(func_tuple.second);
    }
  } else {
    for (size_t i = 0, n = type->numAttributes(); i < n; ++i) {
      setstateTuple(
          module,
          obj->getSlot(i),
          elements,
          qn_cache,
          debug_info_elements,
          debug_handle_manager);
    }
  }
}
} // namespace

void moduleMethodsTuple(
    const Module& module,
    std::vector<c10::IValue>& elements, // note: appended to in-place
    std::vector<c10::IValue>& debug_info_elements,
    BackendDebugHandleManager& debug_handle_manager) {
  auto methods = module.get_methods();
  std::unordered_set<std::string> qn_cache;
  // top level methods
  for (const auto& method : methods) {
    const auto qn = method.function().qualname().qualifiedName();
    if (qn_cache.find(qn) != qn_cache.end()) {
      continue;
    }
    auto func_tuple =
        getFunctionTuple(module, method.function(), debug_handle_manager);
    elements.push_back(func_tuple.first);
    qn_cache.emplace(qn);
    debug_info_elements.push_back(func_tuple.second);
  }

  // __setstate__ of all components
  setstateTuple(
      module,
      module._ivalue(),
      elements,
      qn_cache,
      debug_info_elements,
      debug_handle_manager);
}

void SetExportModuleExtraFilesHook(ExportModuleExtraFilesHook hook) {
  GetExtraFilesHook() = std::move(hook);
}

void SetExportModuleMobileInfoConverter(
    ExportModuleMobileInfoConverter converter) {
  GetMobileInfoConverter() = std::move(converter);
}

class ScriptModuleSerializer {
 public:
  explicit ScriptModuleSerializer(const std::string& filename)
      : writer_(filename) {}

  explicit ScriptModuleSerializer(
      const std::function<size_t(const void*, size_t)>& writer_func)
      : writer_(writer_func) {}

  void serialize(
      const Module& module,
      const ExtraFilesMap& extra_files,
      bool bytecode_format,
      bool save_mobile_debug_info) {
    C10_LOG_API_USAGE_ONCE("torch.script.save");
    writeExtraFiles(module, extra_files);
    // Serialize the model object
    writeArchive("data", module._ivalue());
    // Then we serialize all code info.
    writeCode(module.type());
    // The tensor constants from the code are written to a separate archive
    // so loading the code does not depend on loading the data
    std::vector<IValue> ivalue_constants(
        constant_table_.begin(), constant_table_.end());
    writeArchive(
        kArchiveNameConstants, c10::ivalue::Tuple::create(ivalue_constants));
    if (bytecode_format) {
      writeByteCode(module, save_mobile_debug_info);
      writeMobileMetadata(module, extra_files);
    }

    // Acquires and sets minimum (dynamic) version
    for (auto& item : file_streams_) {
      writer_.setMinVersion(item.value().minVersion());
    }
  }

 private:
  void writeArchive(
      const std::string& archive_name,
      const IValue& value,
      bool use_tensors_archive_table = false) {
    std::vector<char> data;
    // Vector to capture the run-time class types during pickling the IValues
    std::vector<c10::ClassTypePtr> memoizedClassTypes;
    Pickler data_pickle(
        [&](const char* buf, size_t size) {
          data.insert(data.end(), buf, buf + size);
        },
        nullptr,
        [&](const c10::ClassTypePtr& t) {
          return type_name_uniquer_.getUniqueName(t);
        },
        &memoizedClassTypes);
    if (use_tensors_archive_table && !tensors_archive_table_.empty()) {
      data_pickle.updateTensorsArchiveTable(tensors_archive_table_);
    }
    data_pickle.protocol();
    data_pickle.pushIValue(value);
    data_pickle.stop();
    size_t i = 0;
    std::string prefix = archive_name + "/";

    // TODO: currently there exists logic only for archive constant and
    // bytecode, to avoid exporting duplicate tensors. The logic can be more
    // generic such that it can be used by other tensors from other archive, to
    // avoid deduplicating tensors among different archives.

    // Store all tensors from archives `constants` to tensors_archive_table_
    if (archive_name == kArchiveNameConstants) {
      const auto tensor_candidates = data_pickle.tensorData();
      for (size_t tensor_index = 0; tensor_index < tensor_candidates.size();
           tensor_index++) {
        tensors_archive_table_[tensor_candidates[tensor_index]] =
            std::make_pair(kArchiveNameConstants, tensor_index);
      }
    }

    // Export deduplicate tensors only if use_tensors_archive_table is set to
    // true and archive name is `bytecode`
    bool can_use_tensors_archive_table =
        (use_tensors_archive_table && archive_name == kArchiveNameBytecode);

    for (const auto& td : data_pickle.tensorData()) {
      WriteableTensorData writable_td = getWriteableTensorData(td);
      std::string fname = prefix + c10::to_string(i++);
      if (can_use_tensors_archive_table) {
        const auto found = tensors_archive_table_.find(td);
        if (found == tensors_archive_table_.end()) {
          writer_.writeRecord(
              fname, writable_td.data(), writable_td.sizeInBytes());
        }
      } else {
        writer_.writeRecord(
            fname, writable_td.data(), writable_td.sizeInBytes());
      }
    }
    std::string fname = archive_name + ".pkl";
    writer_.writeRecord(fname, data.data(), data.size());

    // serialize all the captured run-time class types
    for (const c10::ClassTypePtr& wroteType : memoizedClassTypes) {
      convertNamedType(wroteType);
    }
  }

  void writeExtraFiles(const Module& module, const ExtraFilesMap& extra_files) {
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

  void writeMobileMetadata(
      const Module& module,
      const ExtraFilesMap& extra_files) {
    auto hook = GetExtraFilesHook();
    auto converter = GetMobileInfoConverter();
    if (!converter) {
      return;
    }
    ExtraFilesMap files_to_write = extra_files;
    // merge hook files and extra files
    if (hook) {
      ExtraFilesMap hook_files = hook(module);
      files_to_write.insert(hook_files.begin(), hook_files.end());
    }
    auto content_to_write = converter(module, files_to_write);
    if (!content_to_write.empty()) {
      writeArchive("metadata", content_to_write);
    }
  }

  void updateSourceRangeTags(const SourceRangeRecords& ranges) {
    for (const auto& range : ranges) {
      if (source_range_tags_.find(range.range) == source_range_tags_.end()) {
        source_range_tags_[range.range] = current_source_range_tag_;
        current_source_range_tag_++;
      }
    }
  }

  void writeCode(const at::NamedTypePtr& root_type) {
    class_deps_.add(root_type);
    for (size_t i = 0; i < class_deps_.size(); ++i) {
      // note: convertNameType may extend class_deps_, so re-checking
      // .size() is necessary
      convertNamedType(class_deps_[i]);
    }

    current_source_range_tag_ = 0;
    // Mapping of filename => src. We need this because multiple classes may go
    // in the same file (e.g. foo.bar.Baz and foo.bar.Qux)
    for (auto& item : file_streams_) {
      const std::string filename = qualifierToArchivePath(item.key(), "code/");

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
      auto range_data = source_range_pickler.pickle(
          item.value().ranges(), source_range_tags_);
      writer_.writeRecord(
          debugFilename,
          range_data.data(),
          range_data.size(),
          range_data.size() > kMinToCompress /*compress*/);
    }
  }

  void writeByteCode(const Module& module, const bool save_mobile_debug_info) {
    std::vector<c10::IValue> elements;
    BackendDebugHandleManager debug_handle_manager;
    elements.emplace_back(
        static_cast<int64_t>(caffe2::serialize::kProducedBytecodeVersion));
    std::vector<c10::IValue> debug_info_elements;
    // Always save debug handles
    debug_info_elements.emplace_back(
        static_cast<int64_t>(caffe2::serialize::kProducedBytecodeVersion));

    moduleMethodsTuple(
        module, elements, debug_info_elements, debug_handle_manager);
    auto telements = Tup(std::move(elements));
    writeArchive("bytecode", telements, false);
    auto debug_info_telements = Tup(std::move(debug_info_elements));

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
      writeArchive("mobile_debug_handles", debug_info_telements);
      // Now get the debug-handles-to-inlined-cs-ptr-map
      // And serialize that in a separate archive
      auto debug_handle_cs_ptr_map = debug_handle_manager.getCallStackPtrMap();
      CallStackDebugInfoPickler cs_debug_info_pickler;
      auto cs_data = cs_debug_info_pickler.pickle(
          debug_handle_cs_ptr_map, source_range_tags_);
      // Write out map: [debug-handle, {source range, InlinedCallStack}]
      std::string filename = "callstack_debug_map.pkl";
      static constexpr size_t kMinToCompress = 200;
      writer_.writeRecord(
          filename,
          cs_data.data(),
          cs_data.size(),
          cs_data.size() > kMinToCompress /*compress*/);
    }
  }

  void convertNamedType(const c10::NamedTypePtr& class_type) {
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

  caffe2::serialize::PyTorchStreamWriter writer_;
  std::vector<at::IValue> constant_table_;

  // key: tensor, value: pair(arhive_name, index)
  TensorIndexMap tensors_archive_table_;

  std::unordered_set<c10::NamedTypePtr> converted_types_;
  PrintDepsTable class_deps_;
  TypeNameUniquer type_name_uniquer_;

  // qualifier, e.g. '__torch__.Bar' -> PythonPrint for the file that will be
  // created
  OrderedDict<std::string, PythonPrint> file_streams_;

  // Uniquely identifies a SourceRange in a model.
  // SourceRanges are associated with Nodes of Graphs.
  // However for mobile deployment we dont intend to ship
  // full JIT with capabilities of reading code and constructing
  // graphs.
  // Instead we serialize the Code generated from graph of the methods.
  // Code is serialized in bytecode format that contains instructions
  // corresponding to the nodes of the graph. Since original graph is gone, the
  // question is how do we identify where the ops, in serialized bytecode, come
  // from in original model code. We do this in two parts.
  // 1. Associate a unique tag to SourceRange.
  // 2. Serialize this unique_tag.
  //  2.1 Meaning save <byte_offset, source_range_tag, source range> instead of
  //      <byte_offset, source range>
  // 3. During serializing model for mobile, i.e. bytecode generation,
  //    save unique tag of SourceRange corresponding to the Node.
  // 4. During deserialization, read all the debug_pkl, to construct a map
  //    of <unique_tag, SourceRange> and use tag saved with OPs in bytecode
  //    to lookup the source range.
  // Strictly speaking we will serialize InlinedCallStack directly, which
  // contains SourceRange. This way we have access to entire callstack and not
  // just source information about where the node is, since bytecode inlines the
  // graph before saving it.
  SourceRangeTagMap source_range_tags_;
  int64_t current_source_range_tag_;
};

void ExportModule(
    const Module& module,
    std::ostream& out,
    const ExtraFilesMap& extra_files,
    bool bytecode_format,
    bool save_mobile_debug_info) {
  ScriptModuleSerializer serializer(
      [&](const void* buf, size_t nbytes) -> size_t {
        out.write(static_cast<const char*>(buf), nbytes);
        return !out ? 0 : nbytes;
      });
  serializer.serialize(
      module, extra_files, bytecode_format, save_mobile_debug_info);
}

void ExportModule(
    const Module& module,
    const std::string& filename,
    const ExtraFilesMap& extra_files,
    bool bytecode_format,
    bool save_mobile_debug_info) {
  ScriptModuleSerializer serializer(filename);
  serializer.serialize(
      module, extra_files, bytecode_format, save_mobile_debug_info);
}

void ExportModule(
    const Module& module,
    const std::function<size_t(const void*, size_t)>& writer_func,
    const ExtraFilesMap& extra_files,
    bool bytecode_format,
    bool save_mobile_debug_info) {
  ScriptModuleSerializer serializer(writer_func);
  serializer.serialize(
      module, extra_files, bytecode_format, save_mobile_debug_info);
}

namespace {
void export_opnames(const script::Module& m, std::set<std::string>& opnames) {
  std::vector<c10::IValue> elements;
  std::vector<c10::IValue> debug_info_elements;
  BackendDebugHandleManager dummy;
  moduleMethodsTuple(m, elements, debug_info_elements, dummy);
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
          op_item.size() == 2,
          "There should be two parts in an operator name.");
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

} // namespace jit
} // namespace torch
