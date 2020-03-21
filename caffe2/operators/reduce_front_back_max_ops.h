#ifndef CAFFE2_OPERATORS_REDUCE_FRONT_BACK_MAX_OPS_H_
#define CAFFE2_OPERATORS_REDUCE_FRONT_BACK_MAX_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context, bool FIRSTDIMS>
class MaxReduceDimsOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit MaxReduceDimsOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        num_reduce_dims_(
            this->template GetSingleArgument<int32_t>("num_reduce_dim", 1)) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() {
    auto& X = Input(0);

    CAFFE_ENFORCE(
        num_reduce_dims_ >= 0 && num_reduce_dims_ <= X.dim(),
        "For N-dim input tensor, support num_reduce_dims in range [0, N].");

    const int rows = FIRSTDIMS ? X.size_to_dim(num_reduce_dims_)
                               : X.size_to_dim(X.dim() - num_reduce_dims_);
    const int cols = FIRSTDIMS ? X.size_from_dim(num_reduce_dims_)
                               : X.size_from_dim(X.dim() - num_reduce_dims_);

    vector<int64_t> output_shape;
    int start_index = FIRSTDIMS ? num_reduce_dims_ : 0;
    int end_index = FIRSTDIMS ? X.dim() : X.dim() - num_reduce_dims_;

    for (int i = start_index; i < end_index; ++i) {
      output_shape.push_back(X.sizes()[i]);
    }
    auto* Y = Output(0, output_shape, at::dtype<float>());
    float* out_data = Y->template mutable_data<float>();

    if (cols == 0 || rows == 0) {
      math::Set(Y->numel(), static_cast<float>(0), out_data, &context_);
      return true;
    }

    const int32_t* lengths_data = nullptr;
    if (InputSize() > 1) {
      const auto& lengths = Input(1);
      lengths_data = lengths.template data<int32_t>();
      CAFFE_ENFORCE(
          num_reduce_dims_ == 1,
          "Given lengths input, the number of reduce dimensions should be one.");
      const int batch_size = FIRSTDIMS ? cols : rows;
      CAFFE_ENFORCE(
          lengths.numel() == batch_size,
          "The size of lengths vector doesn't match the batch size.");
    }

    const float* data = X.template data<float>();
    Compute(rows, cols, data, lengths_data, out_data);
    return true;
  }

 protected:
  void Compute(
      int rows,
      int cols,
      const float* data,
      const int32_t* lengths_data,
      float* out_data);

  int num_reduce_dims_;
};

template <typename T, class Context, bool FIRSTDIMS>
class MaxReduceDimsGradientOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit MaxReduceDimsGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        num_reduce_dims_(
            this->template GetSingleArgument<int32_t>("num_reduce_dim", 1)) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& dY = Input(0);
    auto& X = Input(1);
    auto& Y = Input(2);

    auto* dX = Output(0, X.sizes(), at::dtype<float>());
    const int rows = FIRSTDIMS ? X.size_to_dim(num_reduce_dims_)
                               : X.size_to_dim(X.dim() - num_reduce_dims_);
    const int cols = FIRSTDIMS ? X.size_from_dim(num_reduce_dims_)
                               : X.size_from_dim(X.dim() - num_reduce_dims_);

    const float* dYdata = dY.template data<float>();
    const float* Xdata = X.template data<float>();
    const float* Ydata = Y.template data<float>();

    const int32_t* lengths_data = nullptr;
    if (InputSize() > 3) {
      const auto& lengths = Input(3);
      lengths_data = lengths.template data<int32_t>();
      CAFFE_ENFORCE(
          num_reduce_dims_ == 1,
          "Given lengths input, the number of reduce dimensions should be one.");
      const int batch_size = FIRSTDIMS ? cols : rows;
      CAFFE_ENFORCE(
          lengths.numel() == batch_size,
          "The size of lengths vector doesn't match the batch size.");
    }

    float* dXdata = dX->template mutable_data<float>();
    Compute(rows, cols, dYdata, Xdata, Ydata, lengths_data, dXdata);
    return true;
  }

 protected:
  void Compute(
      int rows,
      int cols,
      const float* dYdata,
      const float* Xdata,
      const float* Ydata,
      const int32_t* lengths_data,
      float* dXdata);

  int num_reduce_dims_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_REDUCE_FRONT_BACK_MAX_OPS_H_
