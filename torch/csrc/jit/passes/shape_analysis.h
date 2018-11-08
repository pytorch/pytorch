#pragma once

#include "torch/csrc/WindowsTorchApiMacro.h"

namespace torch { namespace jit {

struct Graph;
struct CompleteArgumentSpec;
struct ArgumentSpec;

TORCH_API void EraseShapeInformation(Graph & graph);
TORCH_API void PropagateInputShapes(Graph & graph);

}}
