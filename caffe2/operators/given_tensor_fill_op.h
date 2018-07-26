#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/filler_op.h"
#include "caffe2/utils/cast.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class GivenTensorFillOp final : public FillerOp<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  GivenTensorFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<Context>(operator_def, ws) {
    const ArgumentHelper helper(operator_def);
    // GivenTensorFillOp can be provided with a "dtype" arg if float is
    // is specified as T. Otherwise, "dtype" is ignored.
    // In the ideal world, we would get rid of templating of T at all, but we
    // need to provide backwards compatibility.
    if (!std::is_same<T, float>::value || !helper.HasArgument("dtype")) {
      ExtractValues<T>();
    } else {
      auto dtype = cast::GetCastDataType(helper, "dtype");
      switch (dtype) {
        case TensorProto_DataType_FLOAT:
          ExtractValues<float>();
          break;
        case TensorProto_DataType_DOUBLE:
          ExtractValues<double>();
          break;
        case TensorProto_DataType_BOOL:
          ExtractValues<bool>();
          break;
        case TensorProto_DataType_INT32:
          ExtractValues<int>();
          break;
        case TensorProto_DataType_INT64:
          ExtractValues<int64_t>();
          break;
        case TensorProto_DataType_STRING:
          ExtractValues<std::string>();
          break;
        case TensorProto_DataType_UNDEFINED:
          CAFFE_THROW("Cannot have undefined 'dtype' argument");
        default:
          CAFFE_THROW("Unexpected 'dtype' argument value: ", dtype);
      }
    }
  }

  bool Fill(Tensor<Context>* output) override {
    return (this->*body_)(output);
  }

 private:
  template <typename Type>
  void ExtractValues() {
    auto source_values =
        OperatorBase::template GetRepeatedArgument<Type>("values");
    values_.Resize(source_values.size());
    Type* values_data = values_.template mutable_data<Type>();
    for (int i = 0; i < source_values.size(); i++) {
      values_data[i] = static_cast<Type>(source_values[i]);
    }
    body_ = &GivenTensorFillOp::FillWithType<Type>;
  }

  template <typename Type>
  bool FillWithType(Tensor<Context>* output) {
    DCHECK_EQ(output->size(), values_.size())
        << "output size: " << output->size()
        << " given size: " << values_.size();
    auto* data = output->template mutable_data<Type>();
    const Type* values_data = values_.template data<Type>();
    if (output->size()) {
      context_.template Copy<Type, CPUContext, Context>(
          output->size(), values_data, data);
    }
    return true;
  }

  bool (GivenTensorFillOp::*body_)(Tensor<Context>* output);
  TensorCPU values_;
};
} // namespace caffe2
