#pragma once
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <vector>

namespace torch {
namespace jit {

struct Method;
struct Module;
struct PythonPrintImpl;

struct PrintDepsTable {
  void add(const c10::NamedTypePtr& type);

  size_t size() const {
    return table_.size();
  }

  const c10::NamedTypePtr& operator[](size_t index) const {
    return table_[index];
  }

 private:
  std::vector<c10::NamedTypePtr> table_;
  std::unordered_set<c10::NamedTypePtr> non_unique_;
};

struct TORCH_API PythonPrint {
  PythonPrint(
      std::vector<IValue>& constant_table,
      PrintDepsTable& deps_table,
      c10::TypePrinter type_printer = nullptr,
      bool enforce_importable = false);

  void printNamedType(const c10::NamedTypePtr& classType);
  void printFunction(const Function& callee);
  void printMethod(const Function& callee);

  std::string str() const;
  const SourceRangeRecords& ranges() const;
  uint64_t minVersion() const;

 private:
  std::shared_ptr<PythonPrintImpl> pImpl;
};

TORCH_API bool printerHasSpecialCaseFor(c10::Symbol sym);

TORCH_API void jitModuleToPythonCodeAndConstants(
    const Module& module,
    ExtraFilesMap* jit_sources, // output
    std::vector<IValue>* constants // output
);

} // namespace jit
} // namespace torch
