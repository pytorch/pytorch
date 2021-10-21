#pragma once

#include <map>

#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/qualified_name.h>
#include <torch/csrc/jit/backends/backend_debug_handler.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>

namespace torch {
namespace jit {

struct ExportedFunction {
  ExportedFunction(const Function& f, std::unique_ptr<Graph> g, bool t)
      : function(f), optimizedGraph(std::move(g)), toplevel(t) {}
  const Function& function;
  std::unique_ptr<Graph> optimizedGraph;
  bool toplevel;
};

class TORCH_API BytecodeExportSet {
  struct Comparator {
    bool operator()(const c10::QualifiedName& a, const c10::QualifiedName& b)
        const {
      return a.qualifiedName() < b.qualifiedName();
    }
  };

 public:
  BytecodeExportSet() : items_(Comparator{}) {}
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
  std::map<c10::QualifiedName, ExportedFunction, Comparator> items_;
};

} // namespace jit
} // namespace torch
