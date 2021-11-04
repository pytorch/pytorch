#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/filler_op.h"
#include "caffe2/utils/cast.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class GivenTensorByteStringToUInt8FillOp final : public FillerOp<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  explicit GivenTensorByteStringToUInt8FillOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : FillerOp<Context>(operator_def, ws) {
    const ArgumentHelper helper(operator_def);
    if (!helper.HasArgument("dtype")) {
      Extract();
    } else {
      auto dtype = cast::GetCastDataType(helper, "dtype");
      switch (dtype) {
        case TensorProto_DataType_STRING:
          Extract();
          break;
        case TensorProto_DataType_UNDEFINED:
          CAFFE_THROW("Cannot have undefined 'dtype' argument");
        default:
          CAFFE_THROW("Unexpected 'dtype' argument value: ", dtype);
      }
    }
  }

  bool Fill(Tensor* output) override {
    DCHECK_EQ(output->numel(), values_.numel())
        << "output size: " << output->numel()
        << " given size: " << values_.numel();
    auto* data = output->template mutable_data<uint8_t>();
    const uint8_t* values_data = values_.template data<uint8_t>();
    if (output->numel()) {
      context_.template CopySameDevice<uint8_t>(
          output->numel(), values_data, data);
    }
    return true;
  }

 private:
  void Extract() {
    auto source_values = this->template GetRepeatedArgument<string>("values");
    DCHECK_EQ(source_values.size(), 1)
        << "expected size: 1 "
        << " given size: " << source_values.size();

    auto str = source_values[0];
    ReinitializeTensor(
        &values_,
        {static_cast<int64_t>(str.size())},
        at::dtype<uint8_t>().device(CPU));
    uint8_t* values_data = values_.template mutable_data<uint8_t>();
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    for (int i = 0; i < str.size(); i++) {
      values_data[i] = static_cast<uint8_t>(str[i]);
    }
  }

  Tensor values_;
};
} // namespace caffe2
