#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/sgd/weight_scale_op.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(WeightScale, WeightScaleOp<CUDAContext>);

template <typename T>
void weight_scale_update_kernel(
    int N,
    const T* w,
    const T& scale,
    int64_t iter,
    int64_t stepsize,
    int64_t update_upper_bound,
    T* nw,
    CUDAContext* context) {
  const auto w_size = N * sizeof(float);
  if (iter % stepsize != 0 || iter >= update_upper_bound) {
    (void)cudaMemcpy(nw, w, w_size, cudaMemcpyDefault);
  } else {
    // perform the weight scaling
    caffe2::math::Scale<T, T, CUDAContext>(N, scale, w, nw, context);
  }
}

template <>
template <typename T>
bool WeightScaleOp<CUDAContext>::DoRunWithType() {
  const auto iter =
      OperatorBase::Input<Tensor>(ITER, CPU).template data<int64_t>()[0] + 1;
  weight_scale_update_kernel<T>(
      Input(WEIGHTS).size(),
      Input(WEIGHTS).template data<T>(),
      scale_,
      iter,
      stepsize_,
      update_upper_bound_,
      Output(OUTPUT_WEIGHTS)->template mutable_data<T>(),
      &context_);
  return true;
}

} // namespace caffe2
