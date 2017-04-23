#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <
    typename T,
    class Context = CUDAContext,
    bool FIRSTDIMS = true,
    bool NORMALIZE = false>
class ReduceDimsOp : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ReduceDimsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        num_reduce_dims_(
            OperatorBase::GetSingleArgument<int32_t>("num_reduce_dim", 1)) {}

  ~ReduceDimsOp() {}

  bool RunOnDevice() override {
    const auto& input = Input(0);
    const auto* input_data = input.template data<T>();
    auto* Y = Output(0);

    CHECK_LT(num_reduce_dims_, input.dims().size());
    const int M = FIRSTDIMS
        ? input.size_to_dim(num_reduce_dims_)
        : input.size_to_dim(input.ndim() - num_reduce_dims_);
    const int N = FIRSTDIMS
        ? input.size_from_dim(num_reduce_dims_)
        : input.size_from_dim(input.ndim() - num_reduce_dims_);

    vector<TIndex> output_shape;
    int start_index = FIRSTDIMS ? num_reduce_dims_ : 0;
    int end_index = FIRSTDIMS ? input.dims().size()
                              : input.dims().size() - num_reduce_dims_;
    for (int i = start_index; i < end_index; ++i) {
      output_shape.push_back(input.dims()[i]);
    }

    Y->Resize(output_shape);

    int in_dim = FIRSTDIMS ? M : N;

    if (ones_.size() != in_dim) {
      ones_.Resize(in_dim);
      math::Set<T, Context>(
          in_dim,
          static_cast<T>(1),
          ones_.template mutable_data<T>(),
          &context_);
    }

    T alpha = 1.0;
    if (NORMALIZE) { // Static if
      alpha = 1.0 / in_dim;
    }

    math::Gemv<T, Context>(
        FIRSTDIMS ? CblasTrans : CblasNoTrans,
        M,
        N,
        alpha,
        input_data,
        ones_.template data<T>(),
        0.0,
        Y->template mutable_data<T>(),
        &context_);

    return true;
  }

 private:
  Tensor<Context> ones_;
  int num_reduce_dims_;
};

namespace {
template <typename T>
__global__ void
StripedScaleKernel(const int N, const int M, const T* alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, M * N) {
    int k = i / N;
    y[i] = x[i] * alpha[k];
  }
}

namespace {
template <typename T>
__global__ void StripedAxpbyKernel(
    const int N,
    const int M,
    const T a,
    const T* x,
    const T b,
    T* y) {
  CUDA_1D_KERNEL_LOOP(index, N * M) {
    y[index] = x[index % N] * a + y[index] * b;
  }
}
} // namespace
}

template <
    typename T,
    class Context = CUDAContext,
    bool FIRSTDIMS = true,
    bool NORMALIZE = false>
class ReduceDimsGradientOp : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ReduceDimsGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        num_reduce_dims_(
            OperatorBase::GetSingleArgument<int32_t>("num_reduce_dim", 1)) {}

  ~ReduceDimsGradientOp() {}

  bool RunOnDevice() override {
    const auto& grad_in = Input(0);
    const auto& in_shape = Input(1);

    Tensor<CPUContext> shape;
    shape.CopyFrom(in_shape);
    // Copy first dims
    vector<TIndex> output_shape(
        shape.template data<TIndex>(),
        shape.template data<TIndex>() + shape.size());

    auto* out_grad = Output(0);
    out_grad->Resize(output_shape);

    const int M = FIRSTDIMS
        ? out_grad->size_to_dim(num_reduce_dims_)
        : out_grad->size_to_dim(out_grad->ndim() - num_reduce_dims_);
    const int N = FIRSTDIMS
        ? out_grad->size_from_dim(num_reduce_dims_)
        : out_grad->size_from_dim(out_grad->ndim() - num_reduce_dims_);

    int in_dim = FIRSTDIMS ? M : N;

    T alpha = 1.0;
    if (NORMALIZE) { // Static if
      alpha = 1.0 / in_dim;
    }

    math::Set<T, CUDAContext>(
        out_grad->size(),
        FIRSTDIMS ? static_cast<T>(0) : static_cast<T>(alpha),
        out_grad->template mutable_data<T>(),
        &context_);

    if (FIRSTDIMS) {
      StripedAxpbyKernel<T><<<
          CAFFE_GET_BLOCKS(N * M),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          N,
          M,
          alpha,
          grad_in.template data<T>(),
          static_cast<T>(0),
          out_grad->template mutable_data<T>());
    } else {
      StripedScaleKernel<T><<<
          CAFFE_GET_BLOCKS(N * M),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          N,
          M,
          grad_in.template data<T>(),
          out_grad->template data<T>(),
          out_grad->template mutable_data<T>());
    }

    return true;
  }

 private:
  int num_reduce_dims_;
};

REGISTER_CUDA_OPERATOR(ReduceFrontSum, ReduceDimsOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ReduceFrontSumGradient,
    ReduceDimsGradientOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ReduceFrontMean,
    ReduceDimsOp<float, CUDAContext, true, true>);
REGISTER_CUDA_OPERATOR(
    ReduceFrontMeanGradient,
    ReduceDimsGradientOp<float, CUDAContext, true, true>);

REGISTER_CUDA_OPERATOR(ReduceBackSum, ReduceDimsOp<float, CUDAContext, false>);
REGISTER_CUDA_OPERATOR(
    ReduceBackSumGradient,
    ReduceDimsGradientOp<float, CUDAContext, false, false>);

REGISTER_CUDA_OPERATOR(
    ReduceBackMean,
    ReduceDimsOp<float, CUDAContext, false, true>);
REGISTER_CUDA_OPERATOR(
    ReduceBackMeanGradient,
    ReduceDimsGradientOp<float, CUDAContext, false, true>);

} // namespace caffe2
