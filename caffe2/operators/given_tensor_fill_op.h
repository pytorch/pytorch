#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/filler_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class GivenTensorFillOp final : public FillerOp<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  GivenTensorFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<Context>(operator_def, ws) {
    auto source_values =
        OperatorBase::template GetRepeatedArgument<T>("values");
    values_.Resize(source_values.size());
    T* values_data = values_.template mutable_data<T>();
    for (int i = 0; i < source_values.size(); i++) {
      values_data[i] = static_cast<T>(source_values[i]);
    }
  }

  bool Fill(Tensor<Context>* output) override {
    DCHECK_EQ(output->size(), values_.size())
        << "output size: " << output->size()
        << " given size: " << values_.size();
    auto* data = output->template mutable_data<T>();
    const T* values_data = values_.template data<T>();
    if (output->size()) {
      context_.template Copy<T, CPUContext, Context>(
          output->size(), values_data, data);
    }
    return true;
  }

 private:
  TensorCPU values_;
};
}
