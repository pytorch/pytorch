#include "hip/hip_runtime.h"
#include "caffe2/core/hip/common_hip.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/selu_op.h"

namespace caffe2 {
namespace {
template <typename T>
__global__ void SeluKernel(const int N, const T* X, T* Y, T alpha_, T lambda_)
{
    HIP_1D_KERNEL_LOOP(i, N) { Y[i] = lambda_ * (X[i] > 0 ? X[i] : alpha_ * expf(X[i]) - alpha_); }
}

template <typename T>
__global__ void SeluGradientKernel(const int N, const T* Y, const T* dY, T* dX, T alpha_, T lambda_)
{
    const T c = lambda_ * alpha_;
    HIP_1D_KERNEL_LOOP(i, N)
    {
        // Reuse Y[i] to avoid computing exp(X[i])
        dX[i] = Y[i] > 0 ? lambda_ * dY[i] : dY[i] * (Y[i] + c);
    }
}
} // namespace

template <>
bool SeluOp<float, HIPContext>::RunOnDevice()
{
    auto& X = Input(0);
    auto* Y = Output(0);
    CAFFE_ENFORCE_GT(X.size(), 0);
    Y->ResizeLike(X);
    hipLaunchKernelGGL((SeluKernel),
                       dim3(CAFFE_GET_BLOCKS(X.size())),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       static_cast<const int>(X.size()),
                       static_cast<const float*>(X.data<float>()),
                       static_cast<float*>(Y->mutable_data<float>()),
                       static_cast<float>(alpha_),
                       static_cast<float>(lambda_));
    return true;
}

template <>
bool SeluGradientOp<float, HIPContext>::RunOnDevice()
{
    auto& Y  = Input(0);
    auto& dY = Input(1);
    auto* dX = Output(0);
    CAFFE_ENFORCE_GT(Y.size(), 0);
    CAFFE_ENFORCE_EQ(dY.size(), Y.size());
    dX->ResizeLike(Y);
    hipLaunchKernelGGL((SeluGradientKernel),
                       dim3(CAFFE_GET_BLOCKS(Y.size())),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       static_cast<const int>(Y.size()),
                       static_cast<const float*>(Y.data<float>()),
                       static_cast<const float*>(dY.data<float>()),
                       static_cast<float*>(dX->mutable_data<float>()),
                       static_cast<float>(alpha_),
                       static_cast<float>(lambda_));
    return true;
}

REGISTER_HIP_OPERATOR(Selu, SeluOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(SeluGradient, SeluGradientOp<float, HIPContext>);
} // namespace caffe2
