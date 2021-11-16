#pragma once

#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>

namespace torch_lazy_tensors {
namespace compiler {
/**
 * This is a deprecated interface, to be replaced by LazyShapeDtype.cpp and meta kernels.
 *
 * It is only used by legacy ops that haven't been ported yet.
 * */
torch::lazy::Shape InferShape(const torch::lazy::Node* node);

}
}
