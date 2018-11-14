#ifndef CAFFE2_OPERATORS_REDUCE_FRONT_BACK_SUM_MEAN_OPS_H_
#define CAFFE2_OPERATORS_REDUCE_FRONT_BACK_SUM_MEAN_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context, bool FIRSTDIMS, bool NORMALIZE>
class SumReduceDimsOp final : public Operator<Context> {
 public:
  SumReduceDimsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        num_reduce_dims_(
            this->template GetSingleArgument<int32_t>("num_reduce_dim", 1)) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int, int64_t, float, double>>::call(
        this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& X = Input(0);
    auto* Y = Output(0);

    CAFFE_ENFORCE(
        num_reduce_dims_ >= 0 && num_reduce_dims_ <= X.sizes().size(),
        "For N-dim input tensor, support num_reduce_dims in range [0, N].");

    vector<int64_t> output_shape;
    int start_index = FIRSTDIMS ? num_reduce_dims_ : 0;
    int end_index =
        FIRSTDIMS ? X.sizes().size() : X.sizes().size() - num_reduce_dims_;
    for (int i = start_index; i < end_index; ++i) {
      output_shape.push_back(X.sizes()[i]);
    }
    Y->Resize(output_shape);

    const int rows = FIRSTDIMS ? X.size_to_dim(num_reduce_dims_)
                               : X.size_to_dim(X.dim() - num_reduce_dims_);
    const int cols = FIRSTDIMS ? X.size_from_dim(num_reduce_dims_)
                               : X.size_from_dim(X.dim() - num_reduce_dims_);

    const T* in_data = X.template data<T>();
    T* out_data = Y->template mutable_data<T>();

    if (cols == 0 || rows == 0) {
      math::Set(Y->numel(), static_cast<T>(0), out_data, &context_);
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

    Compute(rows, cols, in_data, lengths_data, out_data);

    return true;
  }

 private:
  template <typename T>
  void Compute(
      int rows,
      int cols,
      const T* in_data,
      const int32_t* lengths_data,
      T* out_data);

  int num_reduce_dims_;
};

template <class Context, bool FIRSTDIMS, bool NORMALIZE>
class SumReduceDimsGradientOp final : public Operator<Context> {
 public:
  SumReduceDimsGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        num_reduce_dims_(
            this->template GetSingleArgument<int32_t>("num_reduce_dim", 1)) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int, long, float, double>>::call(
        this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& dY = Input(0);
    auto& input_1 = Input(1);
    auto* dX = Output(0);

    // In previous diff we changed the semantic: Input(1) was changed from
    // the shape of the input to the data tensor. This made the backward
    // computation incompatible with old models. To fix this, we check
    // the dimension and type of Input(1).
    if (input_1.dim() == 1 && input_1.template IsType<int64_t>()) {
      // Input(1) is the shape of the input
      shape_.CopyFrom(input_1);
      // Copy first dims
      vector<int64_t> output_shape(
          shape_.template data<int64_t>(),
          shape_.template data<int64_t>() + shape_.numel());
      dX->Resize(output_shape);
    } else {
      // Input(1) is data tensor X
      dX->ResizeLike(input_1);
    }

    const int rows = FIRSTDIMS ? dX->size_to_dim(num_reduce_dims_)
                               : dX->size_to_dim(dX->dim() - num_reduce_dims_);
    const int cols = FIRSTDIMS
        ? dX->size_from_dim(num_reduce_dims_)
        : dX->size_from_dim(dX->dim() - num_reduce_dims_);

    const int32_t* lengths_data = nullptr;
    if (InputSize() > 2) {
      const auto& lengths = Input(2);
      lengths_data = lengths.template data<int32_t>();
      CAFFE_ENFORCE(
          num_reduce_dims_ == 1,
          "Given lengths input, the number of reduce dimensions should be one.");
      const int batch_size = FIRSTDIMS ? cols : rows;
      CAFFE_ENFORCE(
          lengths.numel() == batch_size,
          "The size of lengths vector doesn't match the batch size.");
    }

    const T* dYdata = dY.template data<T>();
    T* dXdata = dX->template mutable_data<T>();
    Compute<T>(rows, cols, dYdata, lengths_data, dXdata);
    return true;
  }

 private:
  template <typename T>
  void Compute(
      int rows,
      int cols,
      const T* dYdata,
      const int32_t* lengths_data,
      T* dXdata);
  int num_reduce_dims_;
  // scratch space used for former version of this reducer
  Tensor shape_{Context::GetDeviceType()};
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_REDUCE_FRONT_BACK_SUM_MEAN_OPS_H_
