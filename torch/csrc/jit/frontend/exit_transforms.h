
copy: fbcode/caffe2/torch/csrc/jit/frontend/exit_transforms.h
copyrev: fcace1238bace0f6ea3c4dd89e0685bc5d856c0d

#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void TransformExits(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
