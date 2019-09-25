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
    SourceRangeRecords& source_ranges_out,
    const Function& callee,
    bool is_method,
    std::vector<at::Tensor>& tensor_table,
    std::vector<c10::NamedTypePtr>& deps_table,
    bool enforce_importable = false);

TORCH_API void PythonPrint(
    std::ostream& out,
    SourceRangeRecords& source_ranges_out,
    const c10::NamedTypePtr& classType,
    std::vector<at::Tensor>& tensor_table,
    std::vector<c10::NamedTypePtr>& deps_table,
    bool enforce_importable = false);

TORCH_API void LEGACY_PythonPrint(
    std::ostream& out,
    SourceRangeRecords& source_ranges_out,
    const c10::NamedTypePtr& type,
    std::vector<at::Tensor>& tensor_table,
    std::vector<c10::NamedTypePtr>& deps_table,
    bool enforce_importable = false);

TORCH_API void LEGACY_PythonPrint(
    std::ostream& out,
    SourceRangeRecords& source_ranges_out,
    const script::Module& module,
    std::vector<at::Tensor>& tensor_table,
    std::vector<c10::NamedTypePtr>& deps_table,
    bool enforce_importable);

TORCH_API bool printerHasSpecialCaseFor(c10::Symbol sym);
} // namespace jit
} // namespace torch
