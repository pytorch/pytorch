#pragma once

#include "torch/csrc/WindowsTorchApiMacro.h"

namespace torch { namespace jit {
struct Graph;
struct ArgumentSpec;
TORCH_API void PropagateInputShapes(Graph & graph, const ArgumentSpec & spec);

}}
