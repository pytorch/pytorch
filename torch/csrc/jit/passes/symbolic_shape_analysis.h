#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// CAUTION NOT TO BE USED, STILL A WIP, NOT STABLE

TORCH_API void PropagateShapesOnGraph(std::shared_ptr<Graph>& graph);

// don't insert complete tensor shapes in shape compute graphs and instead
// rely on our partial evaluation pipeline to propagate information.
// this is a good proxy for our ability to propagate non-complete shape
// information.
TORCH_API bool setSymbolicShapeAnalysisTestMode(bool value);
TORCH_API bool symbolicShapeAnalysisTestModeEnabled();

} // namespace jit
} // namespace torch
