#pragma once

#include <memory>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch { namespace jit {

struct Graph;

TORCH_API void EraseShapeInformation(std::shared_ptr<Graph> graph);
TORCH_API void PropagateInputShapes(std::shared_ptr<Graph> graph);

}}
