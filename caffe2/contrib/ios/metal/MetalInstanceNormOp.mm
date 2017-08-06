// Copyright 2004-present Facebook. All Rights Reserved.

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#import "MetalCaffeContext.h"
#import "metal_instance_norm.h"

#ifndef CAFFE2_MOBILE
#error "Caffe2 mobile state not defined"
#endif

#if CAFFE2_MOBILE

namespace caffe2 {

class MetalInstanceNormOp final : public Operator<MetalCaffeContext> {
 public:
  MetalInstanceNormOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<MetalCaffeContext>(operator_def, ws),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5)),
        order_(StringToStorageOrder(OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE(epsilon_ >= 0, "Must pass a nonnegative epsilon.");
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW, "Metal only supports NCHW order.");
  }

  bool RunOnDevice() override {
    return RunOnDeviceWithOrderNCHW();
  }

  bool RunOnDeviceWithOrderNCHW();

 protected:
  float        epsilon_;
  StorageOrder order_;
  INPUT_TAGS(INPUT, SCALE, BIAS);
  OUTPUT_TAGS(OUTPUT);
};

bool MetalInstanceNormOp::RunOnDeviceWithOrderNCHW() {
  const auto& X     = Input(INPUT);
  const auto& scale = Input(SCALE);
  const auto& bias  = Input(BIAS);
  auto*       Y     = Output(OUTPUT);

  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);
  Y->ResizeLike(X);

  const auto* Xdata = X.template data<float>();
  const auto* Sdata = scale.template data<uint16_t>();
  const auto* Bdata = bias.template data<uint16_t>();
  auto*       Ydata = Y->template mutable_data<float>();

  id<MTLBuffer> inputDataBuffer  = GetMetalAllocator()->Buffer((void*)Xdata);
  id<MTLBuffer> scaleDataBuffer  = GetMetalAllocator()->Buffer((void*)Sdata);
  id<MTLBuffer> biasDataBuffer   = GetMetalAllocator()->Buffer((void*)Bdata);
  id<MTLBuffer> outputDataBuffer = GetMetalAllocator()->Buffer((void*)Ydata);

  metal_instance_norm(inputDataBuffer, C, H, W, scaleDataBuffer, biasDataBuffer, outputDataBuffer, nil, 0, epsilon_);

  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(MetalInstanceNorm, METAL, MetalInstanceNormOp);
OPERATOR_SCHEMA(MetalInstanceNorm).NumInputs(3, 4).NumOutputs(1, 3).AllowInplace({{0, 0}});

class MetalInstanceNormPReluOp final : public Operator<MetalCaffeContext> {
 public:
  MetalInstanceNormPReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<MetalCaffeContext>(operator_def, ws),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5)),
        order_(StringToStorageOrder(OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE(epsilon_ >= 0, "Must pass a nonnegative epsilon.");
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW, "Metal only supports NCHW order.");
  }

  bool RunOnDevice() override {
    return RunOnDeviceWithOrderNCHW();
  }

  bool RunOnDeviceWithOrderNCHW();

 protected:
  float        epsilon_;
  StorageOrder order_;
  INPUT_TAGS(INPUT, SCALE, BIAS, PRELU);
  OUTPUT_TAGS(OUTPUT);
};

bool MetalInstanceNormPReluOp::RunOnDeviceWithOrderNCHW() {
  const auto& X     = Input(INPUT);
  const auto& scale = Input(SCALE);
  const auto& bias  = Input(BIAS);
  const auto& prelu = Input(PRELU);
  auto*       Y     = Output(OUTPUT);

  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);
  Y->ResizeLike(X);

  const auto* Xdata = X.template data<float>();
  const auto* Sdata = scale.template data<uint16_t>();
  const auto* Bdata = bias.template data<uint16_t>();
  const auto* Pdata = prelu.template data<uint16_t>();
  auto*       Ydata = Y->template mutable_data<float>();

  id<MTLBuffer> inputDataBuffer  = GetMetalAllocator()->Buffer((void*)Xdata);
  id<MTLBuffer> scaleDataBuffer  = GetMetalAllocator()->Buffer((void*)Sdata);
  id<MTLBuffer> biasDataBuffer   = GetMetalAllocator()->Buffer((void*)Bdata);
  id<MTLBuffer> preluDataBuffer  = GetMetalAllocator()->Buffer((void*)Pdata);
  id<MTLBuffer> outputDataBuffer = GetMetalAllocator()->Buffer((void*)Ydata);

  metal_instance_norm(
      inputDataBuffer,
      C,
      H,
      W,
      scaleDataBuffer,
      biasDataBuffer,
      outputDataBuffer,
      preluDataBuffer,
      prelu.size(),
      epsilon_);

  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(MetalInstanceNormPRelu, METAL, MetalInstanceNormPReluOp);
OPERATOR_SCHEMA(MetalInstanceNormPRelu).NumInputs(3, 4).NumOutputs(1, 3).AllowInplace({{0, 0}});
} // namespace caffe2
#endif // CAFFE2_MOBILE
