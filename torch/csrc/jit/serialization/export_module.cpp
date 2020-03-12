#include <torch/csrc/jit/serialization/export.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/serialization/import_export_helpers.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/passes/inliner.h>

#include <caffe2/serialize/inline_container.h>

#include <ATen/ATen.h>

#include <string>
#include <vector>

namespace torch {
namespace jit {

char const * toString(OpCode op);

namespace {
ExportModuleExtraFilesHook& GetExtraFilesHook() {
  static ExportModuleExtraFilesHook func = nullptr;
  return func;
}

static IValue Tup(std::vector<IValue> ivalues) {
  return c10::ivalue::Tuple::create(std::move(ivalues));
}

static IValue Table(const std::vector<std::pair<std::string, IValue>>& entries) {
  std::vector<IValue> ivalue_entries;
  for (const auto& e : entries) {
    ivalue_entries.push_back(Tup({e.first, e.second}));
  }
  return Tup(std::move(ivalue_entries));
}

c10::IValue getFunctionTuple(const Function& func) {
  auto graph = func.graph()->copy();
  Inline(*graph);
  torch::jit::Code code(graph, func.name());

  // operator names
  std::vector<c10::OperatorName> opnames;
  for (size_t i = 0; i < code.instructions().size(); ++i) {
    Instruction ins = code.instructions()[i];
    if (ins.op == OP || ins.op == OPN) {
      auto node = code.instructions_source()[i];
      opnames.emplace_back(node->schema().operator_name());
    }
  }

  // instructions
  std::vector<IValue> instructions;
  instructions.reserve(code.instructions().size());
  for (Instruction ins : code.instructions()) {
    instructions.emplace_back(Tup({toString(ins.op), ins.X, ins.N}));
  }

  // operators
  std::vector<IValue> operators;
  operators.reserve(opnames.size());
  for (const auto& opname : opnames) {
    operators.emplace_back(Tup({opname.name, opname.overload_name}));
  }

  // constants
  const auto& constants = code.constant_table();

  // types
  std::vector<IValue> types;
  types.reserve(code.type_table().size());
  for (const TypePtr& t : code.type_table()) {
    types.emplace_back(t->python_str());
  }

  // since the register location is embedded into the bytecode, pass the register size
  auto register_size = static_cast<int>(code.register_size());

  auto table = Table({{"instructions", Tup(instructions)},
                      {"operators", Tup(operators)},
                      {"constants", Tup(constants)},
                      {"types", Tup(types)},
                      {"register_size", register_size}});

  return Tup({func.qualname().qualifiedName(), table});
}

void setstateTuple(const IValue& ivalue, std::vector<c10::IValue>& elements) {
  if (!ivalue.isObject()) return;
  auto obj = ivalue.toObject();
  auto type = obj->type();
  if (checkHasValidSetGetState(type)) {
    Function *setstate = type->getMethod("__setstate__");
    elements.push_back(getFunctionTuple(*setstate));
  }
  else {
    for (size_t i = 0, n = type->numAttributes(); i < n; ++i) {
      setstateTuple(obj->getSlot(i), elements);
    }
  }
}

void moduleMethodsTuple(const Module& module,
    std::vector<c10::IValue>& elements) {
  auto methods = module.get_methods();
  // top level methods
  for (const auto& method : methods) {
    elements.push_back(getFunctionTuple(method.function()));
  }

  // __setstate__ of all components
  setstateTuple(module._ivalue(), elements);
}

}

void SetExportModuleExtraFilesHook(ExportModuleExtraFilesHook hook) {
  GetExtraFilesHook() = hook;
}

class ScriptModuleSerializer {
 public:
  explicit ScriptModuleSerializer(const std::string& filename)
      : writer_(filename) {}

  explicit ScriptModuleSerializer(
      const std::function<size_t(const void *, size_t)>& writer_func)
      : writer_(writer_func) {}

  void serialize(
      const Module& module,
      const ExtraFilesMap& extra_files,
      bool bytecode_format) {
    C10_LOG_API_USAGE_ONCE("torch.script.save");
    writeExtraFiles(module, extra_files);
    // Serialize the model object
    writeArchive("data", module._ivalue());
    // Then we werialize all code info.
    writeCode(module.type());
    // The tensor constants from the code are written to a separate archive
    // so loading the code does not depend on loading the data
    std::vector<IValue> ivalue_constants(
        constant_table_.begin(), constant_table_.end());
    writeArchive("constants", c10::ivalue::Tuple::create(ivalue_constants));
    if (bytecode_format) {
      writeByteCode(module);
    }
  }

 private:
  void writeArchive(const std::string& archive_name, const IValue& value) {
    std::vector<char> data;
    // Vector to capture the run-time class types during pickling the IValues
    std::vector<c10::ClassTypePtr> memorizedClassTypes;
    Pickler data_pickle(
        [&](const char* buf, size_t size) {
          data.insert(data.end(), buf, buf + size);
        },
        nullptr,
        &memorizedClassTypes);
    data_pickle.protocol();
    data_pickle.pushIValue(value);
    data_pickle.stop();
    size_t i = 0;
    std::string prefix = archive_name + "/";
    for (const auto& td : data_pickle.tensorData()) {
      std::string fname = prefix + c10::to_string(i++);
      writer_.writeRecord(fname, td.data(), td.sizeInBytes());
    }
    std::string fname = archive_name + ".pkl";
    writer_.writeRecord(fname, data.data(), data.size());

    // serialize all the captured run-time class types
    for (const c10::ClassTypePtr& wroteType : memorizedClassTypes) {
      convertNamedType(wroteType);
    }
  }

  void writeExtraFiles(
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
        const std::string key = "extra/" + kv.first;
        writer_.writeRecord(key, kv.second.data(), kv.second.size());
      }
    }
  }

  void writeCode(const at::NamedTypePtr& root_type) {
    class_deps_.push_back(root_type);
    for (size_t i = 0; i < class_deps_.size(); ++i) {
      // note: convertNameType may extend class_deps_, so re-checking
      // .size() is necessary
      convertNamedType(class_deps_[i]);
    }

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
          filename, src.c_str(), src.size(),
          src.size() > kMinToCompress /*compress*/);

      // Write out the debug information
      std::string debugFilename = filename + ".debug_pkl";
      SourceRangePickler source_range_pickler;
      auto range_data =
          source_range_pickler.pickle(item.value().ranges());
      writer_.writeRecord(
          debugFilename,
          range_data.data(),
          range_data.size(),
          range_data.size() > kMinToCompress /*compress*/);
    }
  }

  void writeByteCode(const Module& module) {
    std::vector<c10::IValue> elements;
    moduleMethodsTuple(module, elements);
    auto telements = Tup(std::move(elements));
    writeArchive("bytecode", telements);
  }

  void convertNamedType(const c10::NamedTypePtr& class_type) {
    if (converted_types_.count(class_type)) {
      return;
    }
    converted_types_.insert(class_type);
    std::string qualifier = class_type->name()->prefix();
    PythonPrint* pp = file_streams_.find(qualifier);
    if (!pp) {
      pp = &file_streams_.insert(
          qualifier,
          PythonPrint(
              constant_table_, class_deps_, /*enforce_importable=*/true));
    }
    pp->printNamedType(class_type);
  }

  caffe2::serialize::PyTorchStreamWriter writer_;
  std::vector<at::Tensor> constant_table_;
  std::unordered_set<c10::NamedTypePtr> converted_types_;
  std::vector<c10::NamedTypePtr> class_deps_;

  // qualifier, e.g. '__torch__.Bar' -> PythonPrint for the file that will be
  // created
  OrderedDict<std::string, PythonPrint> file_streams_;
};

void ExportModule(
    const Module& module,
    std::ostream& out,
    const ExtraFilesMap& extra_files,
    bool bytecode_format) {
  ScriptModuleSerializer serializer(
    [&](const void* buf, size_t nbytes) -> size_t {
      out.write(static_cast<const char *>(buf), nbytes);
      return !out ? 0 : nbytes;
    });
  serializer.serialize(module, extra_files, bytecode_format);
}

void ExportModule(
    const Module& module,
    const std::string& filename,
    const ExtraFilesMap& extra_files,
    bool bytecode_format) {
  ScriptModuleSerializer serializer(filename);
  serializer.serialize(module, extra_files, bytecode_format);
}

void ExportModule(
    const Module& module,
    const std::function<size_t(const void*, size_t)>& writer_func,
    const ExtraFilesMap& extra_files,
    bool bytecode_format) {
  ScriptModuleSerializer serializer(writer_func);
  serializer.serialize(module, extra_files, bytecode_format);
}

} // namespace jit
} // namespace torch
