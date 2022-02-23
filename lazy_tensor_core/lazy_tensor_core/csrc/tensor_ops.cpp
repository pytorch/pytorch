#include "lazy_tensor_core/csrc/tensor_ops.h"

#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/util.h>

#include "lazy_tensor_core/csrc/tensor_aten_ops.h"
#include "lazy_tensor_core/csrc/tensor_distributed.h"
#include "lazy_tensor_core/csrc/ts_backend/LazyLazyIr.h"

namespace torch_lazy_tensors {
namespace tensor_ops {

torch::lazy::LazyTensorPtr Select(const torch::lazy::LazyTensorPtr& input, int64_t dim, int64_t index) {
  auto shape = input->shape();
  dim = torch::lazy::GetCanonicalDimensionIndex(dim, shape.Get().dim());
  torch::lazy::LazyTensorPtr result = lazy_tensor_aten_ops::narrow(input, dim, index, 1);
  auto new_dims = torch::lazy::DropDimensions(shape.Get().sizes(), {dim});
  return lazy_tensor_aten_ops::view(result, new_dims);
}

}  // namespace tensor_ops
}  // namespace torch_lazy_tensors
