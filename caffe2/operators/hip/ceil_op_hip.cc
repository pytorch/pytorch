#include "hip/hip_runtime.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/ceil_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T>
__global__ void CeilKernel(const int N, const T* X, T* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = std::ceil(X[i]);
  }
}

template <>
bool CeilOp<float, HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_ENFORCE_GT(X.size(), 0);
  Y->ResizeLike(X);
  hipLaunchKernelGGL((CeilKernel), dim3(CAFFE_GET_BLOCKS(X.size())), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), 
      static_cast<const int>(X.size()), X.data<float>(), Y->mutable_data<float>());
  return true;
}

REGISTER_HIP_OPERATOR(Ceil, CeilOp<float, HIPContext>);
} // namespace caffe2
