// Copyright 2004-present Facebook. All Rights Reserved.

#include "caffe2/core/operator.h"
#import "MetalCaffeContext.h"
#import "metal_prelu.h"

#ifndef CAFFE2_MOBILE
#error "Caffe2 mobile state not defined"
#endif

#if CAFFE2_MOBILE

namespace caffe2 {

class MetalPReluOp final : public Operator<MetalCaffeContext> {
 public:
  MetalPReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<MetalCaffeContext>(operator_def, ws),
        order_(StringToStorageOrder(OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW, "Metal only supports NCHW order.");
  }

  bool RunOnDevice() override;

 protected:
  StorageOrder order_;
};

bool MetalPReluOp::RunOnDevice() {
  const auto& X = Input(0);
  const auto& W = Input(1);
  auto*       Y = Output(0);

  Y->ResizeLike(X);

  const auto* Xdata = X.template data<float>();
  const auto* Wdata = W.template data<uint16_t>();
  auto*       Ydata = Y->template mutable_data<float>();

  id<MTLBuffer> inputDataBuffer  = GetMetalAllocator()->Buffer((void*)Xdata);
  id<MTLBuffer> weightBuffer     = GetMetalAllocator()->Buffer((void*)Wdata);
  id<MTLBuffer> outputDataBuffer = GetMetalAllocator()->Buffer((void*)Ydata);

  metal_prelu(inputDataBuffer, X.dim32(1), X.dim32(3), X.dim32(2), weightBuffer, W.size(), outputDataBuffer);

  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(MetalPRelu, METAL, MetalPReluOp);
OPERATOR_SCHEMA(MetalPRelu).NumInputs(2).NumOutputs(1).AllowInplace({{0, 0}}).IdenticalTypeAndShape();
} // namespace caffe2
#endif // CAFFE2_MOBILE
