#pragma once
#include <c10/util/flat_hash_map.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/ir/scope.h>

namespace torch {
namespace jit {
/*
 * MobileDebugTable:
 * Deserializes debug_pkl and callstack_map records from PT model's zip archive
 * and stores them in a map of debug handles to DebugInfoPair. Debug handles are
 * unique per model and runtime, be in lite interpreter or delegate, an
 * exception of BackendRuntimeException should raised using debug handles.
 * getSourceDebugString method is responsible for translating debug
 * handles to correspond debug information.
 * This debug informatin includes stack trace of model level source code and
 * module hierarchy where the exception occurred.
 */
class MobileDebugTable {
 public:
  MobileDebugTable() = default;
  MobileDebugTable(
      std::unique_ptr<caffe2::serialize::PyTorchStreamReader>& reader,
      const std::shared_ptr<CompilationUnit>& cu);
  std::string getSourceDebugString(
      const int64_t debug_handle,
      const std::string& top_module_type_name = "ModuleTypeUnknown") const;
  std::string getSourceDebugString(
      const std::vector<int64_t>& debug_handles,
      const std::string& top_module_type_name = "ModuleTypeUnknown") const;
  std::string getModuleHierarchyInfo(
      const int64_t debug_handle,
      const std::string& top_module_type_name = "ModuleTypeUnknown") const;
  std::string getModuleHierarchyInfo(
      const std::vector<int64_t>& debug_handles,
      const std::string& top_module_type_name = "ModuleTypeUnknown") const;

 private:
  std::pair<std::string, std::string> getSourceDebugModuleHierarchyInfo(
      const std::vector<int64_t>& debug_handles,
      const std::string& top_module_type_name = "ModuleTypeUnknown") const;
  ska::flat_hash_map<int64_t, DebugInfoTuple> callstack_ptr_map_;
};

} // namespace jit
} // namespace torch
