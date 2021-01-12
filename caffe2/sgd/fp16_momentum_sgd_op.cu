#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"

#include "caffe2/sgd/fp16_momentum_sgd_op.h"

namespace caffe2 {
namespace {

#ifdef __HIPCC__
typedef __half half;
typedef __half2 half2;
#endif

__global__ void FP16MomentumSGDKernel(
    int N,
    const half2* g,
    const half2* m,
    half2* ng,
    half2* nm,
    const float* lr,
    const float mom,
    bool nesterov,
    const float wd,
    half2* param) {
#if __CUDA_ARCH__ >= 530 || defined(__HIP_PLATFORM_HCC__)
  const float lr2 = lr[0];
  const half2 LR = __float2half2_rn(lr2);
  const half2 momentum = __float2half2_rn(mom);
  const half2 weight_decay = __float2half2_rn(wd);

  int n = N / 2;
  if (!nesterov) {
    CUDA_1D_KERNEL_LOOP(i, n) {
      ng[i] = __hfma2(weight_decay, param[i], g[i]);
      const half2 adjusted_gradient =
          __hfma2(LR, ng[i], __hmul2(momentum, m[i]));
      nm[i] = adjusted_gradient;
      ng[i] = adjusted_gradient;
      if (param) {
        param[i] = __hsub2(param[i], ng[i]);
      }

      // odd number of elements
      if (i == 0 && (N % 2)) {
        half *g_half = (half*)g, *param_half = (half*)param, *m_half = (half*)m,
             *nm_half = (half*)nm, *ng_half = (half*)ng;
        ng_half[N - 1] =
            __hfma(__high2half(weight_decay), param_half[N - 1], g_half[N - 1]);
        const half adjusted_gradient_half = __hfma(
            __high2half(LR),
            ng_half[N - 1],
            __hmul(__high2half(momentum), m_half[N - 1]));
        nm_half[N - 1] = adjusted_gradient_half;
        ng_half[N - 1] = adjusted_gradient_half;
        if (param) {
          param_half[N - 1] = __hsub(param_half[N - 1], adjusted_gradient_half);
        }
      }
    }
  } else {
    CUDA_1D_KERNEL_LOOP(i, n) {
      // computing the term (grad + lambda*weight)
      // might need to change in case of denormalization
      ng[i] = __hfma2(weight_decay, param[i], g[i]);
      const half2 mi = m[i];
      const half2 mom_mi = __hmul2(momentum, mi);
      const half2 mi_new = __hfma2(LR, ng[i], mom_mi);
      nm[i] = mi_new;
      ng[i] = __hsub2(__hfma2(mi_new, momentum, mi_new), mom_mi);

      if (param) {
        param[i] = __hsub2(param[i], ng[i]);
      }

      // odd number of elements
      if (i == 0 && (N % 2)) {
        half *g_half = (half*)g, *param_half = (half*)param, *m_half = (half*)m,
             *nm_half = (half*)nm, *ng_half = (half*)ng;
        ng_half[N - 1] =
            __hfma(__high2half(weight_decay), param_half[N - 1], g_half[N - 1]);
        const half mi_half = m_half[N - 1];
        const half mom_mi_half = __hmul(__high2half(momentum), mi_half);
        const half mi_new_half =
            __hfma(__high2half(LR), ng_half[N - 1], mom_mi_half);
        nm_half[N - 1] = mi_new_half;
        ng_half[N - 1] = __hsub(
            __hfma(mi_new_half, __high2half(momentum), mi_new_half),
            mom_mi_half);
        if (param) {
          param_half[N - 1] = __hsub(param_half[N - 1], ng_half[N - 1]);
        }
      }
    }
  }

#else
   CUDA_KERNEL_ASSERT(false);
#endif // CAFFE_HAS_CUDA_FP16
}

__global__ void FP16MomentumSGDFP32Kernel(
    int N,
    const half2* g,
    const half2* m,
    half2* ng,
    half2* nm,
    const float* lr,
    const float mom,
    bool nesterov,
    const float wd,
    half2* param) {
#if __CUDA_ARCH__ >= 530 || defined(__HIP_PLATFORM_HCC__)
  const float lr2 = lr[0];
  const float LR = lr2;
  const float momentum = mom;
  const float weight_decay = wd;

  int n = N / 2;
  if (!nesterov) {
    CUDA_1D_KERNEL_LOOP(i, n) {
      float2 param_float2 = __half22float2(param[i]);
      float2 g_float2 = __half22float2(g[i]);

      float2 ng_float2;
      ng_float2.x = __fmaf_rn(weight_decay, param_float2.x, g_float2.x);
      ng_float2.y = __fmaf_rn(weight_decay, param_float2.y, g_float2.y);

      float2 m_float2 = __half22float2(m[i]);
      float2 adjusted_gradient_float2;
      adjusted_gradient_float2.x =
          __fmaf_rn(LR, ng_float2.x, __fmul_rn(momentum, m_float2.x));
      adjusted_gradient_float2.y =
          __fmaf_rn(LR, ng_float2.y, __fmul_rn(momentum, m_float2.y));

      nm[i] = __float22half2_rn(adjusted_gradient_float2);
      ng[i] = __float22half2_rn(adjusted_gradient_float2);

      if (param) {
        param_float2.x = __fsub_rn(param_float2.x, adjusted_gradient_float2.x);
        param_float2.y = __fsub_rn(param_float2.y, adjusted_gradient_float2.y);
        param[i] = __float22half2_rn(param_float2);
      }
    }
  } else {
    CUDA_1D_KERNEL_LOOP(i, n) {
      // computing the term (grad + lambda*weight)
      // might need to change in case of denormalization

      float2 param_float2 = __half22float2(param[i]);
      float2 g_float2 = __half22float2(g[i]);

      float2 ng_float2;
      ng_float2.x = __fmaf_rn(weight_decay, param_float2.x, g_float2.x);
      ng_float2.y = __fmaf_rn(weight_decay, param_float2.y, g_float2.y);

      const float2 mi_float2 = __half22float2(m[i]);
      float2 mom_mi_float2;
      mom_mi_float2.x = __fmul_rn(momentum, mi_float2.x);
      mom_mi_float2.y = __fmul_rn(momentum, mi_float2.y);
      float2 mi_new_float2;
      mi_new_float2.x = __fmaf_rn(LR, ng_float2.x, mom_mi_float2.x);
      mi_new_float2.y = __fmaf_rn(LR, ng_float2.y, mom_mi_float2.y);

      nm[i] = __float22half2_rn(mi_new_float2);
      ng_float2.x = __fsub_rn(
          __fmaf_rn(mi_new_float2.x, momentum, mi_new_float2.x),
          mom_mi_float2.x);
      ng_float2.y = __fsub_rn(
          __fmaf_rn(mi_new_float2.y, momentum, mi_new_float2.y),
          mom_mi_float2.y);
      ng[i] = __float22half2_rn(ng_float2);

      if (param) {
        param_float2.x = __fsub_rn(param_float2.x, ng_float2.x);
        param_float2.y = __fsub_rn(param_float2.y, ng_float2.y);
        param[i] = __float22half2_rn(param_float2);
      }
    }
  }
#else
   CUDA_KERNEL_ASSERT(false);
#endif // CAFFE_HAS_CUDA_FP16
}
}

template <>
void fp16_momentum_sgd_update<CUDAContext>(
    int N,
    const at::Half* g,
    const at::Half* m,
    at::Half* ng,
    at::Half* nm,
    const float* lr,
    float momentum,
    bool nesterov,
    float weight_decay,
    bool fp32_update,
    at::Half* param,
    CUDAContext* context) {
  const cudaDeviceProp& prop = GetDeviceProperty(0);
  if (prop.major >= kFp16CUDADevicePropMajor) {
    if (!fp32_update) {
      FP16MomentumSGDKernel<<<
          CAFFE_GET_BLOCKS(N / 2),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context->cuda_stream()>>>(
          N,
          reinterpret_cast<const half2*>(g),
          reinterpret_cast<const half2*>(m),
          reinterpret_cast<half2*>(ng),
          reinterpret_cast<half2*>(nm),
          lr,
          momentum,
          nesterov,
          weight_decay,
          reinterpret_cast<half2*>(param));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      // not setting N to N/2
    } else {
      FP16MomentumSGDFP32Kernel<<<
          CAFFE_GET_BLOCKS(N / 2),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context->cuda_stream()>>>(
          N,
          reinterpret_cast<const half2*>(g),
          reinterpret_cast<const half2*>(m),
          reinterpret_cast<half2*>(ng),
          reinterpret_cast<half2*>(nm),
          lr,
          momentum,
          nesterov,
          weight_decay,
          reinterpret_cast<half2*>(param));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      // not setting N to N/2
    }

  } else {
    CAFFE_ENFORCE(false, "FP16MomentumSGDUpdate not supported. Major: ",
      prop.major, " Minor: ", prop.minor);
  }
}

REGISTER_CUDA_OPERATOR(
    FP16MomentumSGDUpdate,
    FP16MomentumSGDUpdateOp<at::Half, CUDAContext>);
OPERATOR_SCHEMA(FP16MomentumSGDUpdate)
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

This op is also functionally equivalent to the FP32MomentumSGDUpdateOp, however
it expects FP16 data and performs its updates in either FP16 precision
(default), or FP32 precision if the 'fp32_update' flag is set to True.

)DOC");
}
