#pragma once

#include "torch/csrc/WindowsTorchApiMacro.h"

namespace torch { namespace jit {

struct Graph;
struct ArgumentSpec;
struct CoarseArgumentSpec;

void EraseShapeInformation(Graph & graph);
TORCH_API void PropagateInputShapes(Graph & graph, const ArgumentSpec & spec);
TORCH_API void PropagateInputShapes(Graph & graph, const CoarseArgumentSpec & spec);

}}
