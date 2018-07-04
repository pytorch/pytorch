#ifndef CAFFE2_OPERATORS_MINMAX_OPS_H_
#define CAFFE2_OPERATORS_MINMAX_OPS_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class MaxMinOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(MaxMinOpBase)

  bool RunOnDevice() override {
    auto& input0 = Input(0);
    auto* output = Output(0);

    output->ResizeLike(input0);
    output->CopyFrom(input0, &context_);

    if (InputSize() == 1) {
      return true;
    }

    // Dimension checking
    for (int i = 1; i < InputSize(); ++i) {
      CAFFE_ENFORCE_EQ(
          output->dims(),
          Input(i).dims(),
          "Description: Input #",
          i,
          ", input dimension:",
          Input(i).dims(),
          " should match output dimension: ",
          output->dims());
    }

    return this->Compute();
  }

  virtual bool Compute() = 0;
};

template <typename T, class Context>
class MaxOp : public MaxMinOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  MaxOp(const OperatorDef& operator_def, Workspace* ws)
      : MaxMinOpBase<T, Context>(operator_def, ws) {}
  virtual ~MaxOp() noexcept {}
  bool Compute() override;
};

template <typename T, class Context>
class SelectGradientOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SelectGradientOpBase)

  bool RunOnDevice() override;
};

template <typename T, class Context>
class MaxGradientOp : public SelectGradientOpBase<T, Context> {
 public:
  MaxGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : SelectGradientOpBase<T, Context>(operator_def, ws) {}
  virtual ~MaxGradientOp() noexcept {}
};

template <typename T, class Context>
class MinOp : public MaxMinOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  MinOp(const OperatorDef& operator_def, Workspace* ws)
      : MaxMinOpBase<T, Context>(operator_def, ws) {}
  virtual ~MinOp() noexcept {}
  bool Compute() override;
};

template <typename T, class Context>
class MinGradientOp : public SelectGradientOpBase<T, Context> {
 public:
  MinGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : SelectGradientOpBase<T, Context>(operator_def, ws) {}
  virtual ~MinGradientOp() noexcept {}
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MINMAX_OPS_H_
