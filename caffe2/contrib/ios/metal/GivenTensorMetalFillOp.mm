// Copyright 2004-present Facebook. All Rights Reserved.

#include "caffe2/operators/filler_op.h"
#import "MetalCaffeContext.h"
#import "data_conversion.h"

namespace caffe2 {

template <typename T1, typename T2>
class GivenTensorMetalFillOp final : public FillerOp<MetalCaffeContext> {
 public:
  GivenTensorMetalFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<MetalCaffeContext>(operator_def, ws) {
    auto source_values = OperatorBase::template GetRepeatedArgument<float>("values");
    for (float f : source_values) {
      values_.push_back(f);
    }
  }

  bool Fill(TensorMetal* output) override {
    DCHECK_EQ(output->size(), values_.size()) << "output size: " << output->size() << " given size: " << values_.size();

    id<MTLBuffer> weightBuffer = GetMetalAllocator()->Buffer((void*)output->template mutable_data<T1>());
    T2* output_data            = (T2*)[weightBuffer contents];

    CAFFE_ENFORCE(output_data != NULL);
    memcpycvt(output_data, values_.data(), output->size());
    return true;
  }

 private:
  vector<float> values_;
};

// uint16_t is used because caffe2 does not support float16_t
REGISTER_CPU_OPERATOR_WITH_ENGINE(GivenTensorFloat16MetalFill, METAL, GivenTensorMetalFillOp<uint16_t, float16_t>);
OPERATOR_SCHEMA(GivenTensorFloat16MetalFill).NumInputs(0, 1).NumOutputs(1).AllowInplace({{0, 0}});
}
