#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/operators/experimental/c10/schemas/layer_norm.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

using caffe2::Tensor;

namespace caffe2 {
namespace {

template <typename T>
void ComputeStdDevAndFusedParams(
    const int N,
    const T* mean,
    const T* var,
    T* stddev,
    T* scale,
    T* bias,
    float epsilon) {
  ConstEigenVectorArrayMap<T> var_arr(var, N);
  EigenVectorArrayMap<T> stddev_arr(stddev, N);
  EigenVectorArrayMap<T> scale_arr(scale, N);
  scale_arr = (var_arr + static_cast<T>(epsilon)).rsqrt();
  stddev_arr = scale_arr * (var_arr + static_cast<T>(epsilon));
  EigenVectorArrayMap<T>(bias, N) =
      -scale_arr * ConstEigenVectorArrayMap<T>(mean, N);
}

template <typename T>
void LayerNormForward(
    const int M,
    const int N,
    const T* X,
    const T* scale,
    const T* bias,
    T* Y) {
  EigenArrayMap<T>(Y, N, M) =
      (ConstEigenArrayMap<T>(X, N, M).rowwise() *
       ConstEigenVectorArrayMap<T>(scale, M).transpose())
          .rowwise() +
      ConstEigenVectorArrayMap<T>(bias, M).transpose();
}

template <class DataType>
void layer_norm_impl(
    const Tensor& X,
    Tensor* Y,
    Tensor* mean,
    Tensor* sig,
    int axis,
    float epsilon,
    ops::LayerNorm::Cache* cache,
    BaseContext* context) {

  CAFFE_ENFORCE_GE(X.ndim(), 2, "LayerNorm requires input dim >= 2.");
  const int canonical_axis = X.canonical_axis_index(axis);
  const int M = X.size_to_dim(canonical_axis);
  const int N = X.size_from_dim(canonical_axis);
  Y->ResizeLike(X);
  std::vector<int> moments_dims(
      X.dims().cbegin(), X.dims().cbegin() + canonical_axis);
  moments_dims.push_back(1);
  mean->Resize(moments_dims);
  sig->Resize(moments_dims);
  cache->scale.Resize(M);
  cache->bias.Resize(M);

  const DataType* X_data = X.template data<DataType>();
  DataType* Y_data = Y->template mutable_data<DataType>();
  DataType* mean_data = mean->template mutable_data<DataType>();
  DataType* sig_data = sig->template mutable_data<DataType>();
  DataType* scale_data = cache->scale.template mutable_data<DataType>();
  DataType* bias_data = cache->bias.template mutable_data<DataType>();

  const std::array<int, 2> dims = {M, N};
  const int axis_ = 1;
  math::Moments<DataType, CPUContext>(
      2, dims.data(), 1, &axis_, X_data, mean_data, sig_data, static_cast<CPUContext*>(context));
  ComputeStdDevAndFusedParams<DataType>(
      M, mean_data, sig_data, sig_data, scale_data, bias_data, epsilon);
  LayerNormForward<DataType>(M, N, X_data, scale_data, bias_data, Y_data);
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
