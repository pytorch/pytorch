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

TORCH_API void PythonPrint(
    std::ostream& out,
    const script::Function& callee,
    bool is_method,
    std::vector<at::Tensor>& tensor_table,
    std::vector<ClassTypePtr>& class_table,
    bool enforce_importable = false);

TORCH_API void PythonPrint(
    std::ostream& out,
    const script::CompilationUnit& cu,
    bool is_method,
    std::vector<at::Tensor>& tensor_table,
    std::vector<ClassTypePtr>& class_table,
    bool enforce_importable);

TORCH_API void PythonPrint(
    std::ostream& out,
    const ClassTypePtr& classType,
    std::vector<at::Tensor>& tensor_table,
    std::vector<ClassTypePtr>& class_table,
    bool enforce_importable = false);

TORCH_API bool printerHasSpecialCaseFor(c10::Symbol sym);
} // namespace jit
} // namespace torch
