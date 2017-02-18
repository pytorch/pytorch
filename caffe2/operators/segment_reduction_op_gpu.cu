#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context = CUDAContext, bool NORMALIZE = false>
class ReduceFrontOp : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ReduceFrontOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        num_reduce_dims_(
            OperatorBase::GetSingleArgument<int32_t>("num_reduce_dim", 1)) {}

  ~ReduceFrontOp() {}

  bool RunOnDevice() override {
    const auto& input = Input(0);
    const auto* input_data = input.template data<T>();
    auto* Y = Output(0);

    CHECK_LT(num_reduce_dims_, input.dims().size());
    const int M = input.size_to_dim(num_reduce_dims_);
    const int N = input.size_from_dim(num_reduce_dims_);

    vector<TIndex> output_shape;
    for (int i = num_reduce_dims_; i < input.dims().size(); ++i) {
      output_shape.push_back(input.dims()[i]);
    }

    Y->Resize(output_shape);

    if (ones_.size() != M) {
      ones_.Resize(M);
      math::Set<T, Context>(
          M, static_cast<T>(1), ones_.template mutable_data<T>(), &context_);
    }

    T alpha = 1.0;
    if (NORMALIZE) { // Static if
      alpha = 1.0 / M;
    }

    math::Gemv<T, Context>(
        CblasTrans,
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

template <typename T, class Context = CUDAContext, bool NORMALIZE = false>
class ReduceFrontGradientOp : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ReduceFrontGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        num_reduce_dims_(
            OperatorBase::GetSingleArgument<int32_t>("num_reduce_dim", 1)) {}

  ~ReduceFrontGradientOp() {}

  bool RunOnDevice() override {
    const auto& grad_in = Input(0);
    const auto& in_shape = Input(1);

    Tensor<CPUContext> shape;
    shape.CopyFrom(in_shape);
    // Copy first dims
    vector<TIndex> output_shape(
        shape.template data<TIndex>(),
        shape.template data<TIndex>() + num_reduce_dims_);
    for (const auto& dim : grad_in.dims()) {
      output_shape.push_back(dim);
    }

    auto* out_grad = Output(0);
    out_grad->Resize(output_shape);

    const int M = out_grad->size_to_dim(num_reduce_dims_);
    const int N = out_grad->size_from_dim(num_reduce_dims_);

    T alpha = 1.0;
    if (NORMALIZE) { // Static if
      alpha = 1.0 / M;
    }

    math::Set<T, CUDAContext>(
        out_grad->size(),
        static_cast<T>(0),
        out_grad->template mutable_data<T>(),
        &context_);

    for (int i = 0; i < M; ++i) {
      math::Axpby<T, CUDAContext>(
          N,
          alpha,
          grad_in.template data<T>(),
          static_cast<T>(0),
          out_grad->template mutable_data<T>() + i * N,
          &context_);
    }

    return true;
  }

 private:
  int num_reduce_dims_;
};

REGISTER_CUDA_OPERATOR(
    ReduceFrontSumGradient,
    ReduceFrontGradientOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ReduceFrontSum, ReduceFrontOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ReduceFrontMean,
    ReduceFrontOp<float, CUDAContext, true>);
REGISTER_CUDA_OPERATOR(
    ReduceFrontMeanGradient,
    ReduceFrontGradientOp<float, CUDAContext, true>);
} // namespace caffe2
