#ifndef CAFFE2_OPERATORS_TT_LINEAR_OP_H_
#define CAFFE2_OPERATORS_TT_LINEAR_OP_H_

#ifdef CAFFE2_USE_MKL
#include <mkl.h>
#endif // CAFFE2_USE_MKL

#include "Eigen/Core"
#include "Eigen/Dense"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context, class Engine = DefaultEngine>
class TTLinearOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  TTLinearOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        inp_sizes_(OperatorBase::GetRepeatedArgument<int>("inp_sizes")),
        out_sizes_(OperatorBase::GetRepeatedArgument<int>("out_sizes")),
        tt_ranks_(OperatorBase::GetRepeatedArgument<int>("tt_ranks")),
        Y_temp_(unique_ptr<Blob>(new Blob())) {}
  ~TTLinearOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(0); // Input array
    const auto& b = Input(1); // Bias array
    const auto& cores = Input(2); // 1D array containing the TT-cores
    auto* Y = Output(0);

    CAFFE_ENFORCE(X.ndim() > 1, "Number of dimensions in X: ", X.ndim());
    CAFFE_ENFORCE(b.ndim() == 1, "Number of dimensions in b: ", b.ndim());
    CAFFE_ENFORCE(
        inp_sizes_.size() == out_sizes_.size(),
        "inp_sizes has size: ",
        inp_sizes_.size(),
        ", out_sizes has size: ",
        out_sizes_.size());
    CAFFE_ENFORCE(
        cores.ndim() == 1, "Number of dimensions in cores: ", cores.ndim());
    // batch size
    const int batch_size = X.ndim() > 1 ? X.dim32(0) : 1;

    // dimension d of tensors
    const int d = inp_sizes_.size();

    // Keep track of index of current core in multiplication
    int cores_idx = 0;

    // Temporary buffer to facilitate multiplication of TT-cores with input
    auto Y_buf = Y_temp_->GetMutable<Tensor<Context>>();
    Y_buf->ResizeLike(X);
    Y_buf->CopyFrom(X);

    // The overall forward pass involves multiplication with each core, where
    // each core has sizes dictated by inp_sizes_ and out_sizes_. Each core thus
    // has size inp_sizes_[i] * tt_ranks_[i] * tt_ranks_[i + 1] * out_sizes_[i].
    for (int i = (d - 1); i >= 0; --i) {
      int curr_rows = inp_sizes_[i] * tt_ranks_[i + 1];
      int curr_cols = tt_ranks_[i] * out_sizes_[i];

      // TODO Replace by Reshape(), once wrappers are written
      Y_buf->Resize(Y_buf->size() / curr_rows, curr_rows);
      Y->Resize(Y_buf->size() / curr_rows, curr_cols);

      // Defensive checks
      CAFFE_ENFORCE(Y_buf->size() % curr_rows == 0, Y_buf->size(), curr_rows);
      CAFFE_ENFORCE(
          cores_idx + curr_rows * curr_cols <= cores.size(),
          cores_idx + curr_rows * curr_cols,
          cores.size());

      // Multiply ith core with the intermediate output
      math::Gemm<float, Context, Engine>(
          CblasNoTrans,
          CblasNoTrans,
          Y_buf->size() / curr_rows,
          curr_cols,
          curr_rows,
          1,
          Y_buf->template data<float>(),
          cores.template data<float>() + cores_idx,
          0,
          Y->template mutable_data<float>(),
          &context_);

      CAFFE_ENFORCE(Y->size() % out_sizes_[i] == 0, Y->size(), out_sizes_[i]);

      // TODO Add GPU support by writing a generic wrapper.
      auto Y_mat = EigenMatrixMap<float>(
          Y->template mutable_data<float>(),
          Y->size() / out_sizes_[i],
          out_sizes_[i]);
      Y_mat = ConstEigenMatrixMap<float>(
                  Y->template data<float>(),
                  out_sizes_[i],
                  Y->size() / out_sizes_[i])
                  .transpose()
                  .eval();

      // Resize operation
      Y_buf->Resize(Y->dim32(0), Y->dim32(1));
      context_.template Copy<float, CPUContext, CPUContext>(
          Y->size(),
          Y->template data<float>(),
          Y_buf->template mutable_data<float>());

      cores_idx += curr_rows * curr_cols;
    }

    // TODO Add GPU support by writing a generic wrapper.
    auto Y_mat = EigenMatrixMap<float>(
        Y->template mutable_data<float>(), batch_size, Y->size() / batch_size);
    Y_mat = ConstEigenMatrixMap<float>(
                Y->template data<float>(), Y->size() / batch_size, batch_size)
                .transpose()
                .eval();
    // TODO Replace by Reshape(), once wrappers are written
    Y->Resize(batch_size, Y->size() / batch_size);

    // Check that output size of Y is the element-wise product of out_sizes
    int prod_out_sizes = 1;
    for (int i = 0; i < out_sizes_.size(); i++) {
      prod_out_sizes *= out_sizes_[i];
    }
    CAFFE_ENFORCE(
        Y->dim32(1) == prod_out_sizes,
        "Output dimension of Y: ",
        Y->dim32(1),
        ", product of out_sizes: ",
        prod_out_sizes);

    // Add bias term
    if (bias_multiplier_.size() != batch_size) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Resize(batch_size);
      math::Set<T, Context>(
          batch_size,
          static_cast<T>(1),
          bias_multiplier_.template mutable_data<T>(),
          &context_);
    }
    math::Gemm<T, Context, Engine>(
        CblasNoTrans,
        CblasNoTrans,
        Y->dim32(0),
        Y->dim32(1),
        1,
        1,
        bias_multiplier_.template data<T>(),
        b.template data<T>(),
        1,
        Y->template mutable_data<T>(),
        &context_);
    return true;
  }

 protected:
  Tensor<Context> bias_multiplier_;
  std::vector<int> inp_sizes_;
  std::vector<int> out_sizes_;
  std::vector<int> tt_ranks_;
  std::unique_ptr<Blob> Y_temp_;
};

// TODO: Complete after verifying utility of TT-layer's forward pass.
template <typename T, class Context, class Engine = DefaultEngine>
class TTLinearGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  TTLinearGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  ~TTLinearGradientOp() {}

  bool RunOnDevice() override {
    return false;
  }

 protected:
  Tensor<Context> bias_multiplier_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_TT_LINEAR_OP_H_
