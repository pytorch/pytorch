// Copyright 2004-present Facebook. All Rights Reserved.

#include "caffe2/core/common.h"

#ifndef CAFFE2_MOBILE
#error "Caffe2 mobile state not defined"
#endif

#if CAFFE2_MOBILE

#include "caffe2/core/operator.h"
#import "data_conversion.h"
#import "MetalCaffeContext.h"
#import "MetalImageFilter.h"
#import "metal_sync_op.h"

namespace caffe2 {
class CopyToMetalGPUOp final : public Operator<MetalCaffeContext> {
 public:
  CopyToMetalGPUOp(const OperatorDef &operator_def, Workspace *ws) : Operator<MetalCaffeContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    const Blob *blob = Inputs()[0];
    const TensorCPU &X = blob->Get<TensorCPU>();

    CAFFE_ENFORCE(X.dim32(0) == 1);
    int input_channels = X.dim32(1);
    int input_width = X.dim32(3);
    int input_height = X.dim32(2);

    auto *Y = Output(0);
    Y->Resize(X.dim32(0), input_channels, input_height, input_width);

    const float *input = X.template data<float>();
    float *output_data = (float *)[GetMetalAllocator()->Buffer((void *)Y->template mutable_data<float>()) contents];

    memcpycvt(output_data, input, Y->size());

    return true;
  }
};
REGISTER_CPU_OPERATOR_WITH_ENGINE(CopyToMetalGPU, METAL, CopyToMetalGPUOp);
OPERATOR_SCHEMA(CopyToMetalGPU).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});

class CopyToMetalGPUFloat16Op final : public Operator<MetalCaffeContext> {
 public:
  CopyToMetalGPUFloat16Op(const OperatorDef &operator_def, Workspace *ws)
      : Operator<MetalCaffeContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    const Blob *blob = Inputs()[0];
    const TensorCPU &X = blob->Get<TensorCPU>();

    auto *Y = Output(0);
    Y->ResizeLike(X);

    const float *input = X.template data<float>();
    float16_t *output_data =
        (float16_t *)[GetMetalAllocator()->Buffer((void *)Y->template mutable_data<uint16_t>()) contents];

    memcpycvt(output_data, input, Y->size());

    return true;
  }
};
REGISTER_CPU_OPERATOR_WITH_ENGINE(CopyToMetalGPUFloat16, METAL, CopyToMetalGPUFloat16Op);
OPERATOR_SCHEMA(CopyToMetalGPUFloat16).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});

class CopyFromMetalGPUOp final : public Operator<CPUContext> {
 public:
  CopyFromMetalGPUOp(const OperatorDef &operator_def, Workspace *ws) : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    const Blob *blob = Inputs()[0];
    const TensorMetal &X = blob->Get<TensorMetal>();

    CAFFE_ENFORCE(X.dim32(0) == 1);
    int input_channels = X.dim32(1);
    int input_width = X.dim32(3);
    int input_height = X.dim32(2);

    auto *Y = Output(0);
    Y->Resize(X.dim32(0), input_channels, input_height, input_width);

    const float *input = (float *)[GetMetalAllocator()->Buffer((void *)X.template data<float>()) contents];
    float *output_data = Y->template mutable_data<float>();

    metal_sync_op();

    memcpycvt(output_data, input, Y->size());

    return true;
  }
};
REGISTER_CPU_OPERATOR(CopyFromMetalGPU, CopyFromMetalGPUOp);
OPERATOR_SCHEMA(CopyFromMetalGPU).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
} // namespace caffe2

#endif // CAFFE2_MOBILE
