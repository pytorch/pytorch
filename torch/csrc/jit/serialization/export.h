#pragma once

#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/jit/serialization/storage_context.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>
#include <ostream>

#ifndef INTERN_DISABLE_ONNX
#include <torch/csrc/jit/serialization/export_onnx.h>
#endif

namespace torch {
namespace jit {

// Serializer for both oldsyle and unified format TorchScript serialization
class TORCH_API ScriptModuleSerializer {
 public:
  explicit ScriptModuleSerializer(
      caffe2::serialize::PyTorchStreamWriter& export_writer)
      : writer_(export_writer), current_source_range_tag_(0) {}

  void writeFiles(const std::string& code_dir);
  void serialize(
      const Module& module,
      const ExtraFilesMap& extra_files,
      bool bytecode_format,
      bool save_mobile_debug_info);
  void serialize_unified_format(Module& module, uint64_t script_module_id);
  SerializationStorageContext& storage_context();

  ~ScriptModuleSerializer() = default;

 private:
  void convertNamedType(const c10::NamedTypePtr& class_type);
  void convertTypes(const at::NamedTypePtr& root_type);
  void writeExtraFiles(const Module& module, const ExtraFilesMap& extra_files);
  void writeByteCode(const Module& module, bool save_mobile_debug_info);
  void writeArchive(
      const IValue& value,
      const std::string& archive_name,
      const std::string& archive_dir,
      const std::string& tensor_dir,
      bool use_storage_context = false,
      bool skip_tensor_data = false);
  void updateSourceRangeTags(const SourceRangeRecords& ranges);

  caffe2::serialize::PyTorchStreamWriter& writer_;
  std::vector<at::IValue> constant_table_;

  std::unordered_set<c10::NamedTypePtr> converted_types_;
  PrintDepsTable class_deps_;
  TypeNameUniquer type_name_uniquer_;
  // qualifier, e.g. '__torch__.Bar' -> PythonPrint for the file that will be
  // created
  OrderedDict<std::string, PythonPrint> file_streams_;
  // Used to keep references of storages around during serialization to solve
  // for ABA memory reuse problem hit when storages are created/destroyed
  // during serialization process. Also used to coordinate sharing of storages
  // between Script and eager modules in torch.package.
  SerializationStorageContext storage_context_;

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

TORCH_API void ExportModule(
    const Module& module,
    std::ostream& out,
    const ExtraFilesMap& metadata = ExtraFilesMap(),
    bool bytecode_format = false,
    bool save_mobile_debug_info = false,
    bool use_flatbuffer = false);

TORCH_API void ExportModule(
    const Module& module,
    const std::string& filename,
    const ExtraFilesMap& metadata = ExtraFilesMap(),
    bool bytecode_format = false,
    bool save_mobile_debug_info = false,
    bool use_flatbuffer = false);

TORCH_API void ExportModule(
    const Module& module,
    const std::function<size_t(const void*, size_t)>& writer_func,
    const ExtraFilesMap& metadata = ExtraFilesMap(),
    bool bytecode_format = false,
    bool save_mobile_debug_info = false,
    bool use_flatbuffer = false);

// Write the bytes of a pickle archive and the tensors referenced inside that
// archive
TORCH_API void writeArchiveAndTensors(
    const std::string& archive_name,
    const char* pickle_bytes,
    size_t size,
    const std::vector<at::Tensor>& tensors,
    caffe2::serialize::PyTorchStreamWriter& out);

// Surrounding system can install an additional hook to produce extra files
// with metadata based on environment every time a module is serialized.
using ExportModuleExtraFilesHook = std::function<ExtraFilesMap(const Module&)>;
TORCH_API void SetExportModuleExtraFilesHook(ExportModuleExtraFilesHook hook);

/**
 * Generates new bytecode for a Script module and returns what the op list
 * would be for a LiteScriptModule based off the current code base. If you
 * have a LiteScriptModule and want to get the currently present
 * list of ops call _export_operator_list instead.
 */
TORCH_API std::vector<std::string> export_opnames(const Module& m);

struct TORCH_API BytecodeEmitMode {
  static bool is_default_value_for_unspecified_arg_enabled();
  static void set_default_value_for_unspecified_arg_enabled(bool enabled);

  static bool is_default_args_before_out_args_enabled();
  static void set_default_args_before_out_args_enabled(bool enabled);

  static bool is_emit_promoted_ops_enabled();
  static void set_default_emit_promoted_ops_enabled(bool enabled);
};

// RAII guard to switch the way JIT emits the bytecode for inputs.
// default_value_for_unspecified_arg:
// true: instruction of default argument values (like LOADC) is emitted.
// false: instruction of default argument values are not emitted. Instead
// they are fetched from operator schema.
// default_args_before_out_args (to forward compatibile support
// operators allowing out arguments and default arguments):
// true: the number of specified arguments will deserialized to (#all_args -
// #default_args). false: the number of specified arguments will deserialized to
// (#all_args).
struct TORCH_API BytecodeEmitModeGuard {
  BytecodeEmitModeGuard(
      bool enable_default_value_for_unspecified_arg,
      bool enable_default_args_before_out_args,
      bool enable_emit_promoted_ops)
      : prev_default_value_for_unspecified_arg_mode(
            BytecodeEmitMode::is_default_value_for_unspecified_arg_enabled()),
        prev_default_args_before_out_args(
            BytecodeEmitMode::is_default_args_before_out_args_enabled()),
        prev_default_emit_promoted_ops(
            BytecodeEmitMode::is_emit_promoted_ops_enabled()) {
    BytecodeEmitMode::set_default_value_for_unspecified_arg_enabled(
        enable_default_value_for_unspecified_arg);
    BytecodeEmitMode::set_default_args_before_out_args_enabled(
        enable_default_args_before_out_args);
    BytecodeEmitMode::set_default_emit_promoted_ops_enabled(
        enable_emit_promoted_ops);
  }
  ~BytecodeEmitModeGuard() {
    BytecodeEmitMode::set_default_value_for_unspecified_arg_enabled(
        prev_default_value_for_unspecified_arg_mode);
    BytecodeEmitMode::set_default_args_before_out_args_enabled(
        prev_default_args_before_out_args);
    BytecodeEmitMode::set_default_emit_promoted_ops_enabled(
        prev_default_emit_promoted_ops);
  }
  bool prev_default_value_for_unspecified_arg_mode;
  bool prev_default_args_before_out_args;
  bool prev_default_emit_promoted_ops;
};

TORCH_API IValue to_tuple(std::vector<IValue> ivalues);
TORCH_API IValue
Table(const std::vector<std::pair<std::string, IValue>>& entries);

// TODO remove these switches once interface call is rolled out.
TORCH_API void enableMobileInterfaceCallExport();
bool getMobileInterfaceCallExport();

TORCH_API CompilationOptions getOptionsFromGlobal();

TORCH_API void save_jit_module(
    const Module& module,
    const std::string& filename,
    const ExtraFilesMap& extra_files = ExtraFilesMap());

TORCH_API DetachedBuffer::UniqueDetachedBuffer save_jit_module_to_bytes(
    const Module& module,
    const ExtraFilesMap& extra_files = ExtraFilesMap());

TORCH_API void save_jit_module_to_write_func(
    const Module& module,
    const ExtraFilesMap& extra_files,
    bool save_mobile_debug_info,
    const std::function<size_t(const void*, size_t)>& writer_func);

} // namespace jit
} // namespace torch
