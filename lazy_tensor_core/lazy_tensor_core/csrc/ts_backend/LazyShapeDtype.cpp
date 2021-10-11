/**
 * This is a handwritten file that accompanies codegenerated header
 * LazyShapeDtype.h
 *
 * The purpose of these shape/dtype inference methods are to fill gaps
 * where we do not yet have structured kernels in pytorch core.  Ops
 * for which there _are_ structured kernels can use meta::op() to infer
 * shape/dtype, and codegen makes use of this.  Ops for which there are not
 * yet structured kernels can still be used with lazy_tensor codegen, but require
 * manual intervention to implement compute_shape_{op} and compute_dtype_{op}.
 *
 */
#include "lazy_tensor_core/csrc/ts_backend/LazyShapeDtype.h"

namespace torch_lazy_tensors{
namespace ir {
namespace ops {

std::vector<std::vector<int64_t>> compute_shape_dropout(const at::Tensor& input, double p, bool train) {
  return {input.sizes().vec()};
}

std::vector<c10::ScalarType> compute_dtype_dropout(const at::Tensor& input, double p, bool train) {
  return {input.scalar_type()};
}

std::vector<std::vector<int64_t>> compute_shape_layer_norm(const at::Tensor& input, at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& bias, double eps, bool cudnn_enable) {
  return {input.sizes().vec()};
}

std::vector<c10::ScalarType> compute_dtype_layer_norm(const at::Tensor& input, at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& bias, double eps, bool cudnn_enable) {
  return {input.scalar_type()};
}

std::vector<std::vector<int64_t>> compute_shape_mean(const at::Tensor& self, c10::optional<at::ScalarType> dtype) {
  return {{}};
}

std::vector<c10::ScalarType> compute_dtype_mean(const at::Tensor& self, c10::optional<at::ScalarType> dtype) {
  if (dtype.has_value()) {
    return {dtype.value()};
  }
  return {self.scalar_type()};
}

std::vector<std::vector<int64_t>> compute_shape_mv(const at::Tensor& self, const at::Tensor& vec) {
  return {{self.size(0)}};
}

std::vector<c10::ScalarType> compute_dtype_mv(const at::Tensor& self, const at::Tensor& vec) {
  return {self.scalar_type()};
}

} // namespace ops
} // namespace ir
} // namespace torch_lazy_tensors
