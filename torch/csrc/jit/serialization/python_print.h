#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>
#include <iostream>
#include <vector>

namespace torch {
namespace jit {

struct Method;
struct Module;
struct PythonPrintImpl;

struct TORCH_API PythonPrint {
  PythonPrint(
      std::vector<IValue>& constant_table,
      std::set<c10::NamedTypePtr>& deps_table,
      c10::TypePrinter type_printer = nullptr,
      bool enforce_importable = false);

  void printNamedType(const c10::NamedTypePtr& classType);
  void printFunction(const Function& callee);
  void printMethod(const Function& callee);

  std::string str() const;
  const SourceRangeRecords& ranges() const;
  uint64_t minVersion() const;

  ~PythonPrint();

 private:
  std::shared_ptr<PythonPrintImpl> pImpl;
};

TORCH_API bool printerHasSpecialCaseFor(c10::Symbol sym);
} // namespace jit
} // namespace torch
