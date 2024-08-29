#pragma once
#include <torch/csrc/Export.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <optional>
#include <vector>

namespace torch {
namespace lazy {

std::optional<SourceLocation> TORCH_PYTHON_API GetPythonFrameTop();

std::vector<SourceLocation> TORCH_PYTHON_API GetPythonFrames();

} // namespace lazy
} // namespace torch
