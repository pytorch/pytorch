// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <unordered_map>
#include <unordered_set>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {
template <class Context>
class CopyRowsToTensorOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CopyRowsToTensorOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    return DispatchHelper<
        TensorTypes<at::Half, float, double, int32_t, int64_t>>::
        call(this, Input(INPUT_TENSOR));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& input_tensor = Input(INPUT_TENSOR);
    auto& indices = Input(INDICES);
    auto& row = Input(ROW);
    auto tensor_width = input_tensor.size(1);
    CAFFE_ENFORCE_EQ(input_tensor.dim(), 2, "INPUT_TENSOR should be 2-d");
    CAFFE_ENFORCE_EQ(indices.dim(), 1, "INDICES should be 1-d");
    CAFFE_ENFORCE_EQ(row.dim(), 1, "ROW should be 1-d");
    CAFFE_ENFORCE_EQ(
        tensor_width,
        row.size(0),
        "width of input tensor should match lengths of row");
    const auto* indices_data = indices.template data<int64_t>();
    const auto* row_data = row.template data<T>();
    auto* output = Output(0);
    auto* output_data = output->template mutable_data<T>();
    CAFFE_ENFORCE(
        IsInputOutputAlias(0, 0), "Input 0 and Output 0 should be alias.");
    for (size_t i = 0; i < indices.sizes()[0]; ++i) {
      std::memcpy(
          output_data + indices_data[i] * tensor_width,
          row_data,
          tensor_width * sizeof(T));
    }
    return true;
  }

 protected:
  INPUT_TAGS(INPUT_TENSOR, INDICES, ROW);
};

template <class Context>
class CopyRowsToTensorGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CopyRowsToTensorGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    return DispatchHelper<
        TensorTypes<at::Half, float, double, int32_t, int64_t>>::
        call(this, Input(0));
  }
  template <typename T>
  bool DoRunWithType() {
    auto* output = Output(0);
    output->ResizeLike(Input(0));
    auto* output_data = output->template mutable_data<T>();
    auto& input = Input(0);
    const auto* input_data = input.template data<T>();
    std::memcpy(output_data, input_data, input.size(0) * sizeof(T));

    return true;
  }
};

} // namespace caffe2
