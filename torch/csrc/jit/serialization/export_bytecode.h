#pragma once

#include <unordered_map>

#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/qualified_name.h>
#include <torch/csrc/jit/backends/backend_debug_handler.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>

namespace torch {
namespace jit {

struct ExportedBytecode {
  ExportedBytecode(
      std::shared_ptr<MobileCode> c,
      const c10::FunctionSchema& s,
      bool t)
      : code(std::move(c)), schema(s), toplevel(t) {}
  std::shared_ptr<MobileCode> code;
  const c10::FunctionSchema& schema;
  bool toplevel;
};

class TORCH_API BytecodeExportSet {
 public:
  using Map = std::unordered_map<c10::QualifiedName, ExportedBytecode>;
  void add(const c10::QualifiedName& qn, ExportedBytecode);
  void update(const c10::QualifiedName& qn, bool toplevel);
  bool contains(const c10::QualifiedName& qn) const;
  void exportIValues(
      std::vector<c10::IValue>&,
      std::vector<c10::IValue>&,
      BackendDebugInfoRecorder&,
      TypeNameUniquer&) const;
  void exportIValues(
      std::vector<c10::IValue>&,
      BackendDebugInfoRecorder&,
      TypeNameUniquer&) const;

 private:
  Map items_;
};

IValue to_tuple(std::vector<IValue> ivalues);
IValue Table(const std::vector<std::pair<std::string, IValue>>& entries);

} // namespace jit
} // namespace torch
