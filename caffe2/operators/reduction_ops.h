#ifndef CAFFE2_OPERATORS_REDUCTION_OPS_H_
#define CAFFE2_OPERATORS_REDUCTION_OPS_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class SumElementsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  explicit SumElementsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        average_(this->template GetSingleArgument<bool>("average", false)) {}
  explicit SumElementsOp(const OperatorDef& operator_def, Workspace* ws, bool average)
      : Operator<Context>(operator_def, ws), average_(average) {}
#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
  explicit SumElementsOp(const c10::FunctionSchema& schema, std::vector<c10::IValue> inputs, std::vector<c10::IValue*> outputs)
      : Operator<Context>(schema, std::move(inputs), std::move(outputs)),
        average_(this->template GetSingleArgument<bool>("average", false)) {}
  explicit SumElementsOp(const c10::FunctionSchema& schema, std::vector<c10::IValue> inputs, std::vector<c10::IValue*> outputs, bool average)
      : Operator<Context>(schema, std::move(inputs), std::move(outputs)), average_(average) {}
#endif
  ~SumElementsOp() {}

  bool RunOnDevice() override {
    auto& X = Input(0);

    auto* sum = Output(0, vector<int64_t>(), at::dtype<T>());

    T* data = sum->template mutable_data<T>();

    math::Sum<T, Context>(
        X.numel(), X.template data<T>(), data, &context_, &scratch_);
    if (average_ && X.numel() > 0) {
      math::Scale<float, T, Context>(
          1,
          static_cast<T>(1.) / X.numel(),
          sum->template data<T>(),
          data,
          &context_);
    }
    return true;
  }

 private:
  bool average_;
  Tensor scratch_{Context::GetDeviceType()};
};

template <typename T, class Context>
class SumElementsIntOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit SumElementsIntOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  ~SumElementsIntOp() {}

  bool RunOnDevice() override {
    auto& X = Input(0);

    auto* sum = Output(0, vector<int64_t>(), at::dtype<T>());
    T* data = sum->template mutable_data<T>();
    math::Sum<T, Context>(
        X.numel(), X.template data<T>(), data, &context_, &scratch_);
    return true;
  }

 private:
  Tensor scratch_{Context::GetDeviceType()};
};

template <typename T, class Context>
class SumElementsGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  explicit SumElementsGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        average_(this->template GetSingleArgument<bool>("average", false)) {}
  explicit SumElementsGradientOp(const OperatorDef& operator_def, Workspace* ws, bool average)
      : Operator<Context>(operator_def, ws), average_(average) {}
#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
  explicit SumElementsGradientOp(const c10::FunctionSchema& schema, std::vector<c10::IValue> inputs, std::vector<c10::IValue*> outputs)
      : Operator<Context>(schema, std::move(inputs), std::move(outputs)),
        average_(this->template GetSingleArgument<bool>("average", false)) {}
  explicit SumElementsGradientOp(const c10::FunctionSchema& schema, std::vector<c10::IValue> inputs, std::vector<c10::IValue*> outputs, bool average)
      : Operator<Context>(schema, std::move(inputs), std::move(outputs)), average_(average) {}
#endif
  ~SumElementsGradientOp() {}

  bool RunOnDevice() override;

 private:
  bool average_;
};

template <class Context>
class SumSqrElementsOp : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(SumSqrElementsOp)
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    bool average = this->template GetSingleArgument<bool>("average", false);
    auto& X = Input(0);

    auto* sum = Output(0, vector<int64_t>(), at::dtype<T>());
    math::SumSqr<T, Context>(
        X.numel(),
        X.template data<T>(),
        sum->template mutable_data<T>(),
        &context_,
        &scratch_);
    if (average && X.numel() > 0) {
      math::Scale<float, T, Context>(
          1,
          float(1.) / X.numel(),
          sum->template data<T>(),
          sum->template mutable_data<T>(),
          &context_);
    }
    return true;
  }

 private:
  Tensor scratch_{Context::GetDeviceType()};
};

template <typename T, class Context, bool ROWWISE>
class MaxReductionOp : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(MaxReductionOp)
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& X = Input(0);
    CAFFE_ENFORCE_EQ(X.dim(), 3);

    const int batch_size = X.dim32(0);
    const int M = X.dim32(1);
    const int N = X.dim32(2);

    auto* Y = Output(0, {batch_size, ROWWISE ? M : N}, at::dtype<T>());

    if (ROWWISE) {
      math::RowwiseMax<T, Context>(
          batch_size * M,
          N,
          X.template data<T>(),
          Y->template mutable_data<T>(),
          &context_);
    } else {
      const int input_size = N * M;
      for (int i = 0; i < batch_size; ++i) {
        math::ColwiseMax<T, Context>(
            M,
            N,
            X.template data<T>() + i * input_size,
            Y->template mutable_data<T>() + i * N,
            &context_);
      }
    }
    return true;
  }
};

template <typename T, class Context, bool ROWWISE>
class MaxReductionGradientOp : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(MaxReductionGradientOp)
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;
};

} // namespace caffe2

#endif
