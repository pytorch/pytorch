#ifndef CAFFE2_OPERATORS_MINMAX_OPS_H_
#define CAFFE2_OPERATORS_MINMAX_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/operators/elementwise_ops_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class MaxOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  USE_SIMPLE_CTOR_DTOR(MaxOp)

  bool RunOnDevice() override {
    const auto& X0 = Input(0);
    const T* X0_data = X0.template data<T>();
    const int N = X0.numel();

    if (InputSize() == 1) {
      auto* Y = Output(0);
      Y->ResizeLike(X0);
      if (Y != &X0) {
        T* Y_data = Y->template mutable_data<T>();
        context_.template CopySameDevice<T>(N, X0_data, Y_data);
      }
    }

    else {
      std::vector<std::vector<int>> input_dims_list;
      for (int i = 0; i < InputSize(); i++) {
        std::vector<int> input_dims(
            Input(i).sizes().cbegin(), Input(i).sizes().cend());
        input_dims_list.push_back(input_dims);
      }
      std::vector<int> output_dims_int =
          elementwise_ops_utils::ComputeBroadcastDims(input_dims_list);

      std::vector<int64_t> output_dims = std::vector<int64_t>(
          output_dims_int.cbegin(), output_dims_int.cend());

      auto* output = Output(0, output_dims, at::dtype<T>());

      math::Broadcast(
          input_dims_list[0].size(),
          input_dims_list[0].data(),
          output_dims_int.size(),
          output_dims_int.data(),
          T(1),
          X0.template data<T>(),
          output->template mutable_data<T>(),
          &context_);

      std::unique_ptr<caffe2::Blob> inputBlob =
          caffe2::make_unique<caffe2::Blob>();
      auto* mutTensor = BlobGetMutableTensor(inputBlob.get(), caffe2::CPU);
      mutTensor->Resize(output_dims);
      auto outData = mutTensor->template mutable_data<T>();

      for (int i = 1; i < InputSize(); ++i) {
        math::Broadcast(
            input_dims_list[i].size(),
            input_dims_list[i].data(),
            output_dims_int.size(),
            output_dims_int.data(),
            T(1),
            Input(i).template data<T>(),
            outData,
            &context_);

        math::Max<T, Context>(
            N,
            outData,
            output->template data<T>(),
            output->template mutable_data<T>(),
            &context_);
      }
    }
    return true;
  }
};

template <typename T, class Context>
class MinOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  USE_SIMPLE_CTOR_DTOR(MinOp)

  bool RunOnDevice() override {
    const auto& X0 = Input(0);
    auto* Y = Output(0);
    Y->ResizeLike(X0);
    const T* X0_data = X0.template data<T>();
    T* Y_data = Y->template mutable_data<T>();
    const int N = X0.numel();
    if (InputSize() == 1) {
      if (Y != &X0) {
        context_.template CopySameDevice<T>(N, X0_data, Y_data);
      }
      return true;
    }
    const auto& X1 = Input(1);
    CAFFE_ENFORCE_EQ(
        X0.sizes(),
        Y->sizes(),
        "Description: Input #1, input dimension:",
        X1.sizes(),
        " should match output dimension: ",
        Y->sizes());
    const T* X1_data = X1.template data<T>();
    math::Min<T, Context>(N, X0_data, X1_data, Y_data, &context_);
    for (int i = 2; i < InputSize(); ++i) {
      const auto& Xi = Input(i);
      CAFFE_ENFORCE_EQ(
          Xi.sizes(),
          Y->sizes(),
          "Description: Input #",
          i,
          ", input dimension:",
          Input(i).sizes(),
          " should match output dimension: ",
          Y->sizes());
      const T* Xi_data = Xi.template data<T>();
      math::Min<T, Context>(N, Y_data, Xi_data, Y_data, &context_);
    }
    return true;
  }
};

template <typename T, class Context>
class SelectGradientOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SelectGradientOpBase)

  bool RunOnDevice() override;
};

template <typename T, class Context>
class MaxGradientOp final : public SelectGradientOpBase<T, Context> {
 public:
  MaxGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : SelectGradientOpBase<T, Context>(operator_def, ws) {}

  ~MaxGradientOp() = default;
};

template <typename T, class Context>
class MinGradientOp final : public SelectGradientOpBase<T, Context> {
 public:
  MinGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : SelectGradientOpBase<T, Context>(operator_def, ws) {}

  ~MinGradientOp() = default;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MINMAX_OPS_H_
