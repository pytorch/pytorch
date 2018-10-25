#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/operators/experimental/c10/schemas/layer_norm.h"
#include "caffe2/utils/eigen_utils.h"

using caffe2::Tensor;

namespace {
template <typename T>
using EigenMatrixMapRowMajor = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename T>
using ConstEigenMatrixMapRowMajor = Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
} // namespace

namespace caffe2 {
namespace {

template <class DataType>
void layer_norm_impl(
    const Tensor& input,
    Tensor* output,
    Tensor* mean,
    Tensor* stdev,
    int axis,
    float epsilon,
    ops::LayerNorm::Cache* cache) {
  CAFFE_ENFORCE_GE(input.dims().size(), 2, "LayerNorm requires input dim >= 2");

  const auto canonical_axis = input.canonical_axis_index(axis);
  const int left = input.size_to_dim(canonical_axis);
  const int right = input.size_from_dim(canonical_axis);

  output->ResizeLike(input);
  std::vector<int64_t> stats_dims(
      input.dims().begin(), input.dims().begin() + canonical_axis);
  stats_dims.push_back(1);
  mean->Resize(stats_dims);
  stdev->Resize(stats_dims);

  auto input_map = ConstEigenMatrixMapRowMajor<float>(
      input.template data<float>(), left, right);
  auto mean_map = EigenMatrixMapRowMajor<float>(
      mean->template mutable_data<float>(), left, 1);
  auto stdev_map = EigenMatrixMapRowMajor<float>(
      stdev->template mutable_data<float>(), left, 1);
  auto output_map = EigenMatrixMapRowMajor<float>(
      output->template mutable_data<float>(), left, right);

  auto sqr = [](float f) { return f * f; };
  auto add_ep = [epsilon](float f) { return f + epsilon; };
  auto fsqrt = [](float f) { return std::sqrt(f); };
  // Calculate row-wise statistics
  mean_map = input_map.rowwise().mean();
  stdev_map =
      (input_map.unaryExpr(sqr).rowwise().mean() - mean_map.unaryExpr(sqr))
          .unaryExpr(add_ep)
          .unaryExpr(fsqrt);
  output_map = (input_map - mean_map.replicate(1, right))
                   .cwiseQuotient(stdev_map.replicate(1, right));
}
} // namespace
} // namespace caffe2

namespace c10 {
C10_REGISTER_KERNEL(caffe2::ops::LayerNorm)
    .kernel(&caffe2::layer_norm_impl<float>)
    .dispatchKey(c10::DispatchKey<1>{
        c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU,
                                                 LayoutId(0),
                                                 caffe2::TypeMeta::Id<float>()}});
} // namespace c10
