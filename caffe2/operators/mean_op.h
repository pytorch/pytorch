#ifndef CAFFE2_OPERATORS_MEAN_OPS_H_
#define CAFFE2_OPERATORS_MEAN_OPS_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

template <class Context>
class MeanOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(MeanOp)

  template <typename T>
  bool DoRunWithType() {
    auto& input0 = Input(0);
    auto* output = Output(0);

    output->ResizeLike(input0);
    output->CopyFrom(input0, &context_);

    if (InputSize() == 1) {
      return true;
    }

    // Dimension checking
    for (int i = 1; i < InputSize(); ++i) {
      if (output->dims() != Input(i).dims()) {
        CAFFE_THROW(
            "Check failed: output->dims() == Input(i).dims().",
            "Description: Input #",
            i,
            ", input dimension:",
            Input(i).dims(),
            " should match output dimension: ",
            output->dims());
      }
    }

    T* output_data = output->template mutable_data<T>();
    for (int i = 1; i < InputSize(); ++i) {
      math::Add(
          output->size(),
          output_data,
          Input(i).template data<T>(),
          output_data,
          &context_);
    }

    math::Scale(
        output->size(),
        1.0f / InputSize(),
        output_data,
        output_data,
        &context_);

    return true;
  }

  bool RunOnDevice() override {
    if (Input(0).template IsType<float>()) {
      return DoRunWithType<float>();
    } else {
      CAFFE_THROW(
          "Mean operator only supports 32-bit float, but",
          " input was of type ",
          Input(0).meta().name());
    }
  }
};

template <class Context>
class MeanGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  MeanGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  template <typename T>
  bool DoRunWithType() {
    auto& dY = Input(0);
    const auto* dY_data = dY.template data<T>();
    int size = dY.size();

    int num_inputs = OutputSize();
    float scale = 1.0f / num_inputs;

    // dX0 = scale * dY
    auto* dX0 = Output(0);
    dX0->ResizeLike(dY);
    math::Scale(
        size, scale, dY_data, dX0->template mutable_data<T>(), &context_);

    // Copy the rest dX
    for (int i = 1; i < num_inputs; i++) {
      auto* cur_dX = Output(i);
      cur_dX->ResizeLike(dY);
      cur_dX->CopyFrom(*dX0, &context_);
    }

    return true;
  }

  bool RunOnDevice() override {
    if (Input(0).template IsType<float>()) {
      return DoRunWithType<float>();
    } else {
      CAFFE_THROW(
          "Mean operator only supports 32-bit float, but",
          " input was of type ",
          Input(0).meta().name());
    }
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MEAN_OPS_H_
