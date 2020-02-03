#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

/*
 * Since ONNX has no concept of an exception need
 * to remove them prior to generating ONNX graph
 */
void FixupJitExceptions(std::shared_ptr<Graph>& graph);

}
} // namespace torch
