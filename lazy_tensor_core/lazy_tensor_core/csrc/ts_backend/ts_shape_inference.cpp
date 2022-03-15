#include <ATen/Tensor.h>
#include <ATen/core/Reduction.h>
#include <ATen/native/ConvUtils.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/ts_backend/ops/cast.h>
#include <torch/csrc/lazy/ts_backend/ops/device_data.h>
#include <torch/csrc/lazy/ts_backend/ops/expand.h>
#include <torch/csrc/lazy/ts_backend/ops/scalar.h>
#include <torch/csrc/lazy/core/permutation_util.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>
#include <torch/jit.h>

#include "lazy_tensor_core/csrc/ops/squeeze.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_backward.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_forward.h"
#include "lazy_tensor_core/csrc/ops/unsqueeze.h"

namespace torch_lazy_tensors {
namespace compiler {

torch::lazy::Shape InferShape(const torch::lazy::Node* node) {
  LOG(FATAL) << "InferShape no longer implemented.";
}
}  // namespace compiler
}  // namespace torch_lazy_tensors
