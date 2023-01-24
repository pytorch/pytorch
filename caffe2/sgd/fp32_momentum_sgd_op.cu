#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"

#include "caffe2/sgd/fp32_momentum_sgd_op.h"

namespace caffe2 {
namespace {

__global__ void FP32MomentumSGDKernel(
    int N,
    const float2* g,
    const float2* m,
    float2* ng,
    float2* nm,
    const float* lr,
    const float mom,
    bool nesterov,
    const float wd,
    float2* param) {
#if __CUDA_ARCH__ >= 530
  const float lr2 = lr[0];
  const float LR = lr2;
  const float momentum = mom;
  const float weight_decay = wd;

  int n = N / 2;
  if (!nesterov) {
    CUDA_1D_KERNEL_LOOP(i, n) {
      ng[i].x = __fmaf_rn(weight_decay, param[i].x, g[i].x);
      ng[i].y = __fmaf_rn(weight_decay, param[i].y, g[i].y);

      float2 mi_float2 = m[i];
      float2 adjusted_gradient_float2;
      adjusted_gradient_float2.x =
          __fmaf_rn(LR, ng[i].x, __fmul_rn(momentum, mi_float2.x));
      adjusted_gradient_float2.y =
          __fmaf_rn(LR, ng[i].y, __fmul_rn(momentum, mi_float2.y));

      nm[i] = adjusted_gradient_float2;
      ng[i] = adjusted_gradient_float2;

      if (param) {
        param[i].x = __fsub_rn(param[i].x, adjusted_gradient_float2.x);
        param[i].y = __fsub_rn(param[i].y, adjusted_gradient_float2.y);
      }
    }
  } else {
    CUDA_1D_KERNEL_LOOP(i, n) {
      // computing the term (grad + lambda*weight)
      // might need to change in case of denormalization

      ng[i].x = __fmaf_rn(weight_decay, param[i].x, g[i].x);
      ng[i].y = __fmaf_rn(weight_decay, param[i].y, g[i].y);

      const float2 mi_float2 = m[i];
      float2 mom_mi_float2;
      mom_mi_float2.x = __fmul_rn(momentum, mi_float2.x);
      mom_mi_float2.y = __fmul_rn(momentum, mi_float2.y);
      float2 mi_new_float2;
      mi_new_float2.x = __fmaf_rn(LR, ng[i].x, mom_mi_float2.x);
      mi_new_float2.y = __fmaf_rn(LR, ng[i].y, mom_mi_float2.y);

      nm[i] = mi_new_float2;
      ng[i].x = __fsub_rn(
          __fmaf_rn(mi_new_float2.x, momentum, mi_new_float2.x),
          mom_mi_float2.x);
      ng[i].y = __fsub_rn(
          __fmaf_rn(mi_new_float2.y, momentum, mi_new_float2.y),
          mom_mi_float2.y);

      if (param) {
        param[i].x = __fsub_rn(param[i].x, ng[i].x);
        param[i].y = __fsub_rn(param[i].y, ng[i].y);
      }
    }
  }
#else
   CUDA_KERNEL_ASSERT(false);
#endif // CAFFE_HAS_CUDA_FP16
}
}

template <>
void fp32_momentum_sgd_update<CUDAContext>(
    int N,
    const float* g,
    const float* m,
    float* ng,
    float* nm,
    const float* lr,
    float momentum,
    bool nesterov,
    float weight_decay,
    float* param,
    CUDAContext* context) {
  FP32MomentumSGDKernel<<<
      CAFFE_GET_BLOCKS(N / 2),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(
      N,
      reinterpret_cast<const float2*>(g),
      reinterpret_cast<const float2*>(m),
      reinterpret_cast<float2*>(ng),
      reinterpret_cast<float2*>(nm),
      lr,
      momentum,
      nesterov,
      weight_decay,
      reinterpret_cast<float2*>(param));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // not setting N to N/2
  // TODO_ check float performance vs float2
}

REGISTER_CUDA_OPERATOR(
    FP32MomentumSGDUpdate,
    FP32MomentumSGDUpdateOp<float, CUDAContext>);
OPERATOR_SCHEMA(FP32MomentumSGDUpdate)
    .NumInputs(4)
    .NumOutputs(3)
    .AllowInplace({{0, 0}, {1, 1}, {3, 2}})
    .TensorInferenceFunction([](const OperatorDef& /* unused */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(3);
      out[0] = in[0];
      out[1] = in[1];
      out[2] = in[3];
      return out;
    })
    .SetDoc(R"DOC(

Computes the momentum SGD update similarly to the MomentumSGDUpdateOp,
however this op also performs the weight decay update at the same time, thus
making it more efficient.

This op is also functionally equivalent to the FP16MomentumSGDUpdateOp, however
it expects FP32 data and performs its updates in FP32 precision.

)DOC");
}
