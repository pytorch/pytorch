#include "caffe2/operators/softsign_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {

template <typename T>
inline __host__ __device__ T SquareHIP(const T x) {
  return x * x;
}

template <typename T>
inline __device__ T hip_abs(T x);
template <>
inline __device__ float hip_abs(float x)
{
    return fabsf(x);
}
template <>
inline __device__ double hip_abs(double x)
{
    return fabs(x);
}

template <typename T>
__global__ void SoftsignHIPKernel(const int N, const T* X, T* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = __ldg(X + i) / (T(1) + hip_abs(__ldg(X + i)));
  }
}

template <typename T>
__global__ void
SoftsignGradientHIPKernel(const int N, const T* dY, const T* X, T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
    dX[i] = __ldg(dY + i) / SquareHIP(T(1) + hip_abs(__ldg(X + i)));
  }
}

} // namespace

template <>
template <typename T>
bool SoftsignFunctor<HIPContext>::
operator()(const int N, const T* X, T* Y, HIPContext* context) const {
  hipLaunchKernelGGL(SoftsignHIPKernel<T>, dim3(CAFFE_GET_BLOCKS(static_cast<const int>(N))), dim3(CAFFE_HIP_NUM_THREADS), 0, context->hip_stream(), static_cast<const int>(N), X, Y);
  return true;
}

template <>
template <typename T>
bool SoftsignGradientFunctor<HIPContext>::Forward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& /* X_dims */,
    const T* dY,
    const T* X,
    T* dX,
    HIPContext* context) const {
  const int size = std::accumulate(
      dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
  hipLaunchKernelGGL(SoftsignGradientHIPKernel<T>, dim3(CAFFE_GET_BLOCKS(static_cast<const int>(size))), dim3(CAFFE_HIP_NUM_THREADS), 0, context->hip_stream(), static_cast<const int>(size), dY, X, dX);
  return true;
}

REGISTER_HIP_OPERATOR(
    Softsign,
    UnaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        SoftsignFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    SoftsignGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        SoftsignGradientFunctor<HIPContext>>);

} // namespace caffe2
