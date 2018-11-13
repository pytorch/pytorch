#ifndef CAFFE2_OPERATORS_INT8_GIVEN_TENSOR_FILL_OP_H_
#define CAFFE2_OPERATORS_INT8_GIVEN_TENSOR_FILL_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/filler_op.h"
#include "caffe2/utils/cast.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace int8 {

class Int8GivenTensorFillOp final : public Operator<CPUContext> {
 public:
  Int8GivenTensorFillOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        scale_(this->template GetSingleArgument<float>("Y_scale", 1.0)),
        zero_point_(
            this->template GetSingleArgument<int32_t>("Y_zero_point", 0)),
        shape_(this->template GetRepeatedArgument<int64_t>("shape")) {
    ExtractValues();
  }

  bool RunOnDevice() override {
    auto* output = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    output->t.Resize(shape_);
    output->scale = scale_;
    output->zero_point = zero_point_;
    return Fill(output);
  }

 private:
  void ExtractValues() {
    auto source_values = this->template GetSingleArgument<string>("values", "");
    values_.Resize(source_values.size());
    uint8_t* values_data = values_.template mutable_data<uint8_t>();
    for (int i = 0; i < source_values.size(); i++) {
      values_data[i] = static_cast<uint8_t>(source_values[i]);
    }
  }

  bool Fill(Int8TensorCPU* output) {
    DCHECK_EQ(output->t.numel(), values_.numel())
        << "output size: " << output->t.numel()
        << " given size: " << values_.numel();
    auto* data = output->t.template mutable_data<uint8_t>();
    const uint8_t* values_data = values_.template data<uint8_t>();
    if (output->t.numel()) {
      context_.template CopySameDevice<uint8_t>(
          output->t.numel(), values_data, data);
    }
    return true;
  }

  float scale_;
  int32_t zero_point_;
  vector<int64_t> shape_;
  Tensor values_{CPU};
};

class Int8GivenIntTensorFillOp final : public Operator<CPUContext> {
 public:
  Int8GivenIntTensorFillOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        scale_(this->template GetSingleArgument<float>("Y_scale", 1.0)),
        zero_point_(
            this->template GetSingleArgument<int32_t>("Y_zero_point", 0)),
        shape_(this->template GetRepeatedArgument<int64_t>("shape")) {
    ExtractValues();
  }

  bool RunOnDevice() override {
    auto* output = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    output->t.Resize(shape_);
    output->scale = scale_;
    output->zero_point = zero_point_;
    return Fill(output);
  }

 private:
  void ExtractValues() {
    auto source_values = this->template GetRepeatedArgument<int32_t>("values");
    values_.Resize(source_values.size());
    auto* values_data = values_.template mutable_data<int32_t>();
    for (int i = 0; i < source_values.size(); i++) {
      values_data[i] = static_cast<int32_t>(source_values[i]);
    }
  }

  bool Fill(Int8TensorCPU* output) {
    DCHECK_EQ(output->t.numel(), values_.numel())
        << "output size: " << output->t.numel()
        << " given size: " << values_.numel();
    auto* data = output->t.template mutable_data<int32_t>();
    const auto* values_data = values_.template data<int32_t>();
    if (output->t.numel()) {
      context_.template CopySameDevice<int32_t>(
          output->t.numel(), values_data, data);
    }
    return true;
  }

  float scale_;
  int32_t zero_point_;
  vector<int64_t> shape_;
  Tensor values_{CPU};
};

} // namespace int8
} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_GIVEN_TENSOR_FILL_OP_H_
