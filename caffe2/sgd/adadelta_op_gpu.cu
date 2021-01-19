#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/sgd/adadelta_op.h"

namespace caffe2 {

namespace {

__global__ void AdadeltaUpdateKernel(
    int N,
    const float* w,
    const float* g,
    const float* h,
    const float* d,
    const float epsilon,
    const float decay,
    const float* lr,
    float* nw,
    float* nh,
    float* nd) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float di = d[i];
    float hi = nh[i] = decay * h[i] + (1.0f - decay) * gi * gi;
    float ng = sqrtf(di + epsilon) * rsqrtf(hi + epsilon) * gi;
    nw[i] = w[i] + lr[0] * ng;
    nd[i] = decay * di + (1.0f - decay) * ng * ng;
  }
}

template <>
void AdadeltaUpdate<CUDAContext>(
    int N,
    const float* w,
    const float* g,
    const float* h,
    const float* d,
    const float epsilon,
    const float decay,
    const float* lr,
    float* nw,
    float* nh,
    float* nd,
    CUDAContext* context) {
  AdadeltaUpdateKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(N, w, g, h, d, epsilon, decay, lr, nw, nh, nd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

template <typename SIndex, typename THalf>
__global__ void SparseAdadeltaKernel(
    const size_t N,
    const size_t grad_slice_sz,
    const float epsilon,
    const float decay,
    const SIndex* indices,
    const float* grad,
    const float* lr,
    THalf* param,
    THalf* param_mom,
    THalf* param_mom_delta) {
  const float LR = lr[0];
  CUDA_1D_KERNEL_LOOP(i, N) {
    const size_t gradIdx = i;
    const SIndex index = indices[i / grad_slice_sz];
    const size_t paramIdx = index * grad_slice_sz + (i % grad_slice_sz);

    float mom_new = decay * param_mom[paramIdx] +
        (1.0f - decay) * grad[gradIdx] * grad[gradIdx];
    param_mom[paramIdx] = mom_new;
    float grad_new = sqrtf(epsilon + param_mom_delta[paramIdx]) *
        rsqrtf(mom_new + epsilon) * grad[gradIdx];
    float param_new = LR * grad_new + param[paramIdx];
    param[paramIdx] = param_new;
    float mom_delta_new = decay * param_mom_delta[paramIdx] +
        (1.0f - decay) * grad_new * grad_new;
    param_mom_delta[paramIdx] = mom_delta_new;
  }
}

template <class Context>
class CUDASparseAdadeltaOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CUDASparseAdadeltaOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(float, "epsilon", epsilon_, 1e-5f),
        OP_SINGLE_ARG(float, "decay", decay_, 0.95f) {}

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(PARAM).size(), Input(MOMENT_GRAD).size());
    CAFFE_ENFORCE_EQ(Input(PARAM).size(), Input(MOMENT_DELTA).size());
    CAFFE_ENFORCE_EQ(Input(LR).size(), 1);
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).ndim()));

    // Enforce domain constraints on attributes
    CAFFE_ENFORCE_GE(epsilon_, 0.0f);
    CAFFE_ENFORCE_GT(decay_, 0.0f);
    CAFFE_ENFORCE_LT(decay_, 1.0f);

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename IndexType>
  bool DoRunWithType() {
    auto n = Input(INDICES).size();
    if (n == 0) {
      return true;
    }
    return DispatchHelper<TensorTypes2<float, at::Half>, IndexType>::call(
        this, Input(PARAM));
  }

  template <typename IndexType, typename THalf>
  bool DoRunWithType2() {
    const auto* lr = Input(LR).template data<float>();
    const auto* indices = Input(INDICES).template data<IndexType>();
    const auto* gradIn = Input(GRAD).template data<float>();
    const auto* paramIn = Input(PARAM).template data<THalf>();
    const auto* momentIn = Input(MOMENT_GRAD).template data<THalf>();
    const auto* momentDeltaIn = Input(MOMENT_DELTA).template data<THalf>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<THalf>();
    auto* momentOut =
        Output(OUTPUT_MOMENT_GRAD)->template mutable_data<THalf>();
    auto* momentDeltaOut =
        Output(OUTPUT_MOMENT_DELTA)->template mutable_data<THalf>();

    auto N = Input(GRAD).size();
    auto grad_slice_sz = Input(GRAD).size_from_dim(Input(INDICES).ndim());
    if (N == 0) {
      // empty grad, nothing to do here, not even launching the kernel
      return true;
    }
    SparseAdadeltaKernel<IndexType, THalf>
        <<<CAFFE_GET_BLOCKS(N),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            N,
            grad_slice_sz,
            epsilon_,
            decay_,
            indices,
            gradIn,
            lr,
            paramOut,
            momentOut,
            momentDeltaOut);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
  }

 protected:
  const float epsilon_;
  const float decay_;
  INPUT_TAGS(PARAM, MOMENT_GRAD, MOMENT_DELTA, INDICES, GRAD, LR);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_GRAD, OUTPUT_MOMENT_DELTA);
};

REGISTER_CUDA_OPERATOR(Adadelta, AdadeltaOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(SparseAdadelta, CUDASparseAdadeltaOp<CUDAContext>);
} // namespace caffe2
