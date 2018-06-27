#include "hip/hip_runtime.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/floor_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T>
__global__ void FloorKernel(const int N, const T* X, T* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = std::floor(X[i]);
  }
}

template <>
bool FloorOp<float, HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_ENFORCE_GT(X.size(), 0);
  Y->ResizeLike(X);
  hipLaunchKernelGGL((FloorKernel), dim3(CAFFE_GET_BLOCKS(X.size())), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), 
      static_cast<const int>(X.size()), X.data<float>(), Y->mutable_data<float>());
  return true;
}

REGISTER_HIP_OPERATOR(Floor, FloorOp<float, HIPContext>);

} // namespace caffe2
