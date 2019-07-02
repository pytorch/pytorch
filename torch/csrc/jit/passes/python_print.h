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

// A pair of (byte offset, SourceRange) describing a specific segment
// of the output stream
struct TaggedRange {
  TaggedRange(size_t bytes, SourceRange range)
      : bytes(bytes), range(std::move(range)) {}
  size_t bytes;
  SourceRange range;
};
using SourceRangeRecords = std::vector<TaggedRange>;

TORCH_API void PythonPrint(
    std::ostream& out,
    SourceRangeRecords& source_ranges_out,
    const Function& callee,
    bool is_method,
    std::vector<at::Tensor>& tensor_table,
    std::vector<c10::NamedTypePtr>& class_table,
    bool enforce_importable = false);

TORCH_API void PythonPrint(
    std::ostream& out,
    SourceRangeRecords& source_ranges_out,
    const script::CompilationUnit& cu,
    bool is_method,
    std::vector<at::Tensor>& tensor_table,
    std::vector<c10::NamedTypePtr>& class_table,
    bool enforce_importable);

TORCH_API void PythonPrint(
    std::ostream& out,
    SourceRangeRecords& source_ranges_out,
    const c10::NamedTypePtr& classType,
    std::vector<at::Tensor>& tensor_table,
    std::vector<c10::NamedTypePtr>& class_table,
    bool enforce_importable = false);

TORCH_API bool printerHasSpecialCaseFor(c10::Symbol sym);
} // namespace jit
} // namespace torch
