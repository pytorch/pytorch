#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <iostream>
#include <string>
#include <vector>

namespace torch {
namespace lazy {

c10::optional<SourceLocation> TORCH_API GetPythonFrameTop();

std::vector<SourceLocation> TORCH_API GetPythonFrames();

std::ostream& TORCH_API operator<<(std::ostream& stream,
                         const std::vector<SourceLocation>& frames);

}  // namespace lazy
}  // namespace torch
