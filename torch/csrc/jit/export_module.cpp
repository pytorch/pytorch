#include <torch/csrc/jit/export.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/import_export_helpers.h>
#include <torch/csrc/jit/passes/python_print.h>
#include <torch/csrc/jit/pickle.h>
#include <torch/csrc/jit/source_range_serialization.h>
#include <torch/csrc/jit/instruction.h>
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
};
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
      const script::Module& module,
      const script::ExtraFilesMap& extra_files,
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
      std::string fname = prefix + std::to_string(i++);
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
      const script::Module& module,
      const script::ExtraFilesMap& extra_files) {
    // Write out extra files.
    for (const auto& kv : extra_files) {
      const std::string key = "extra/" + kv.first;
      writer_.writeRecord(key, kv.second.data(), kv.second.size());
    }
    auto hook = GetExtraFilesHook();
    if (hook) {
      script::ExtraFilesMap hook_files = hook(module);
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

    // Mapping of filename => src. We need this because multiple clases may go
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

  void writeByteCode(const script::Module& module) {
    auto methods = module.get_methods();
    std::vector<c10::IValue> elements;
    for (const auto& method : methods) {
      const auto& func = method.function();
      auto graph = func.graph()->copy();
      Inline(*graph);
      torch::jit::Code code(graph);
      // Make a copy of opnames. Some of them may be changed for mobile later.
      std::vector<c10::OperatorName> opnames;
      for (size_t i = 0; i < code.instructions().size(); ++i) {
        Instruction ins = code.instructions()[i];
        if (ins.op == OP) {
          auto node = code.instructions_source()[i];
          opnames.emplace_back(node->schema().operator_name());
        }
      }

      // instructions
      std::vector<IValue> inss;
      for (size_t i = 0; i < code.instructions().size(); ++i) {
        Instruction ins = code.instructions()[i];
        TORCH_CHECK(isOpSupportedInMobile(ins.op), toString(ins.op),
                    " is not supported in mobile module.");
        if (ins.op == OP) {
          if (opnames[ins.X].name == "prim::ListConstruct" ||
              opnames[ins.X].name == "prim::TupleConstruct" ||
              opnames[ins.X].name == "prim::TupleUnpack" ||
              opnames[ins.X].name == "aten::format") {
            auto node = code.instructions_source()[i];
            ins.op = OPN;
            if (opnames[ins.X].name == "prim::TupleUnpack") {
              ins.N = node->outputs().size();
            } else {
              ins.N = node->inputs().size();
            }
            if (opnames[ins.X].name == "prim::ListConstruct") {
              ListTypePtr lt = node->output()->type()->expect<ListType>();
              if (lt->getElementType() == IntType::get()) {
                opnames[ins.X].overload_name = "int";
              } else if (lt->getElementType() == FloatType::get()) {
                opnames[ins.X].overload_name = "float";
              } else if (lt->getElementType() == BoolType::get()) {
                opnames[ins.X].overload_name = "bool";
              } else if (lt->getElementType()->isSubtypeOf(TensorType::get())) {
                opnames[ins.X].overload_name = "Tensor";
              } else {
                opnames[ins.X].overload_name = "generic";
              }
            } else if (opnames[ins.X].name == "prim::TupleConstruct" &&
                       node->output()->type()->expect<TupleType>()->name().has_value()) {
              AT_WARN("Named tuple is serialized as un-named tuple.");
            }
          }
        }
        std::vector<IValue> insv{toString(ins.op), ins.X, ins.N};
        inss.emplace_back(c10::ivalue::Tuple::create(std::move(insv)));
      }
      auto instructions = c10::ivalue::Tuple::create(std::move(inss));
      auto named_ins = c10::ivalue::Tuple::create({"instructions", instructions});

      // operators
      std::vector<IValue> opss;
      for (const auto& opname : opnames) {
        opss.emplace_back(c10::ivalue::Tuple::create({opname.name, opname.overload_name}));
      }
      auto operators = c10::ivalue::Tuple::create(std::move(opss));
      auto named_ops = c10::ivalue::Tuple::create({"operators", operators});

      // constants
      auto constants = c10::ivalue::Tuple::create(code.constant_table());
      auto named_consts = c10::ivalue::Tuple::create({"constants", constants});

      // since the register location is embedded into the bytecode, pass the register size
      auto named_regsize = c10::ivalue::Tuple::create({"register_size",
                                                       static_cast<int>(code.register_size())});

      auto element = c10::ivalue::Tuple::create({named_ins, named_ops, named_consts, named_regsize});
      elements.push_back(c10::ivalue::Tuple::create({func.qualname().qualifiedName(), element}));
    }
    auto telements = c10::ivalue::Tuple::create(std::move(elements));
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
      pp->LEGACY_printOpVersion();
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
  bool bytecode_format_;
};

void ExportModule(
    const script::Module& module,
    std::ostream& out,
    const script::ExtraFilesMap& extra_files,
    bool bytecode_format) {
  ScriptModuleSerializer serializer(
    [&](const void* buf, size_t nbytes) -> size_t {
      out.write(static_cast<const char *>(buf), nbytes);
      return !out ? 0 : nbytes;
    });
  serializer.serialize(module, extra_files, bytecode_format);
}

void ExportModule(
    const script::Module& module,
    const std::string& filename,
    const script::ExtraFilesMap& extra_files,
    bool bytecode_format) {
  ScriptModuleSerializer serializer(filename);
  serializer.serialize(module, extra_files, bytecode_format);
}

void ExportModule(
    const script::Module& module,
    const std::function<size_t(const void*, size_t)>& writer_func,
    const script::ExtraFilesMap& extra_files,
    bool bytecode_format) {
  ScriptModuleSerializer serializer(writer_func);
  serializer.serialize(module, extra_files, bytecode_format);
}

} // namespace jit
} // namespace torch
