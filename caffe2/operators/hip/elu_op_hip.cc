#include "caffe2/operators/elu_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void EluHIPKernel(const int N, const T alpha, const T* X, T* Y);

template <>
__global__ void
EluHIPKernel<float>(const int N, const float alpha, const float* X, float* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] =
        __ldg(X + i) < 0 ? alpha * (expf(__ldg(X + i)) - 1.0f) : __ldg(X + i);
  }
}

template <typename T>
__global__ void EluGradientHIPKernel(
    const int N,
    const T alpha,
    const T* dY,
    const T* Y,
    T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
    dX[i] = __ldg(Y + i) < 0 ? __ldg(dY + i) * (__ldg(Y + i) + alpha)
                             : __ldg(dY + i);
  }
}

} // namespace

template <>
template <typename T>
bool EluFunctor<HIPContext>::
operator()(const int N, const T* X, T* Y, HIPContext* context) const {
  hipLaunchKernelGGL(EluHIPKernel<T>, dim3(CAFFE_GET_BLOCKS(static_cast<const int>(N))), dim3(CAFFE_HIP_NUM_THREADS), 0, context->hip_stream(), static_cast<const int>(N), alpha, X, Y);
  return true;
}

template <>
template <typename T>
bool EluGradientFunctor<HIPContext>::Forward(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const T* Y,
    const T* dY,
    T* dX,
    HIPContext* context) const {
  const int size = std::accumulate(
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
  hipLaunchKernelGGL(EluGradientHIPKernel<T>, dim3(CAFFE_GET_BLOCKS(static_cast<const int>(size))), dim3(CAFFE_HIP_NUM_THREADS), 0, context->hip_stream(), static_cast<const int>(size), alpha, dY, Y, dX);
  return true;
}

REGISTER_HIP_OPERATOR(
    Elu,
    UnaryElementwiseWithArgsOp<
        TensorTypes<float>,
        HIPContext,
        EluFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    EluGradient,
    BinaryElementwiseWithArgsOp<
        TensorTypes<float>,
        HIPContext,
        EluGradientFunctor<HIPContext>>);

} // namespace caffe2
