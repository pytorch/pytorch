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
      std::vector<at::Tensor>& tensor_table,
      std::vector<c10::NamedTypePtr>& deps_table,
      bool enforce_importable = false);

  void printNamedType(const c10::NamedTypePtr& classType);
  void printFunction(const Function& callee);
  void printMethod(const Function& callee);

  std::string str() const;
  const SourceRangeRecords& ranges() const;

  ~PythonPrint();

  void LEGACY_printOpVersion();

 private:
  std::shared_ptr<PythonPrintImpl> pImpl;
};

TORCH_API bool printerHasSpecialCaseFor(c10::Symbol sym);
} // namespace jit
} // namespace torch
