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

struct ExportedFunction {
  ExportedFunction(
      const Module& m,
      const Function& f,
      std::unique_ptr<Graph> g,
      bool t)
      : mod(m), function(f), optimizedGraph(std::move(g)), toplevel(t) {}
  Module mod;
  const Function& function;
  std::unique_ptr<Graph> optimizedGraph;
  bool toplevel;
};

class TORCH_API BytecodeExportSet {
 public:
  BytecodeExportSet() = default;
  BytecodeExportSet(const BytecodeExportSet&) = delete;
  BytecodeExportSet& operator=(const BytecodeExportSet&) = delete;
  BytecodeExportSet(BytecodeExportSet&&) = default;
  BytecodeExportSet& operator=(BytecodeExportSet&&) = default;

  void add(const c10::QualifiedName& qn, ExportedFunction);
  void update(const c10::QualifiedName& qn, bool toplevel);
  bool contains(const c10::QualifiedName& qn) const;

  template <typename F>
  void visit(F&& f) {
    for (auto& item : items_) {
      if (item.second.toplevel) {
        f(item.first, item.second);
      }
    }
    for (auto& item : items_) {
      if (!item.second.toplevel) {
        f(item.first, item.second);
      }
    }
  }

 private:
  std::unordered_map<c10::QualifiedName, ExportedFunction> items_;
};

} // namespace jit
} // namespace torch
