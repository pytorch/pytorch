#pragma once
#include <c10/util/Optional.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <vector>

namespace torch {
namespace lazy {

c10::optional<SourceLocation> TORCH_PYTHON_API GetPythonFrameTop();

std::vector<SourceLocation> TORCH_PYTHON_API GetPythonFrames();

} // namespace lazy
} // namespace torch
