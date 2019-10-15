#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir.h>
#include <iostream>
#include <vector>

namespace torch {
namespace jit {

namespace script {
struct Method;
struct Module;
} // namespace script

struct PythonPrintImpl;

struct TORCH_API PythonPrint {
  PythonPrint(
      std::ostream& out,
      SourceRangeRecords& source_ranges_out,
      std::vector<at::Tensor>& tensor_table,
      std::vector<c10::NamedTypePtr>& deps_table,
      bool enforce_importable = false);

  void printNamedType(const c10::NamedTypePtr& classType);
  void printFunction(const Function& callee);
  void printMethod(const Function& callee);
  // must be called exactly once!
  void finish();
  ~PythonPrint();

 private:
  std::unique_ptr<PythonPrintImpl> pImpl;
};

TORCH_API bool printerHasSpecialCaseFor(c10::Symbol sym);
} // namespace jit
} // namespace torch
