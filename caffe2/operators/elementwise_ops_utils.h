#ifndef CAFFE2_OPERATORS_ELEMENTWISE_OPS_UTILS_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_OPS_UTILS_H_

#include <tuple>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {
namespace elementwise_ops_utils {

TORCH_API std::tuple<size_t, size_t, size_t>
ComputeLegacyBroadcastSizes(const Tensor& A, const Tensor& B, int axis);

TORCH_API std::vector<int> ComputeBinaryBroadcastForwardDims(
    const c10::ArrayRef<int>& A_dims,
    const c10::ArrayRef<int>& B_dims);

TORCH_API void ComputeBinaryBroadcastBackwardAxes(
    const std::vector<int>& A_dims,
    const std::vector<int>& B_dims,
    std::vector<int>* A_axes,
    std::vector<int>* B_axes);

TORCH_API void ComputeBinaryBroadcastBackwardDims(
    const std::vector<int>& A_dims,
    const std::vector<int>& B_dims,
    std::vector<int>* A_back_dims,
    std::vector<int>* B_back_dims);

} // namespace elementwise_ops_utils
} // namespace caffe2

#endif // CAFFE2_OPERATORS_ELEMENTWISE_OPS_UTILS_H_
