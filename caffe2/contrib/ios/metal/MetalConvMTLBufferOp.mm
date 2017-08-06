// Copyright 2004-present Facebook. All Rights Reserved.

#include "caffe2/core/common.h"

#ifndef CAFFE2_MOBILE
#error "Caffe2 mobile state not defined"
#endif

#if CAFFE2_MOBILE

#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/conv_transpose_unpool_op_base.h"
#import "metal_convolution.h"
#import "MetalCaffeContext.h"

namespace caffe2 {

class MetalConvMTLBufferOp final : public ConvPoolOpBase<MetalCaffeContext> {
 public:
  MetalConvMTLBufferOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<MetalCaffeContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW, "Metal only supports NCHW order.");
  }

  bool RunOnDeviceWithOrderNCHW() override;

  const TensorCPU& InputFromTensorCPU(int idx) {
    const Blob*                           blob = Inputs()[idx];
    return blob->Get<TensorCPU>();
  }

  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

bool MetalConvMTLBufferOp::RunOnDeviceWithOrderNCHW() {
  const TensorMetal& X      = Input(INPUT);
  auto&              filter = Input(FILTER);
  auto&              bias   = InputFromTensorCPU(BIAS);
  TensorMetal*       Y      = Output(0);

  CAFFE_ENFORCE(X.ndim() == 4, "Input dim should be 4");
  const int N = X.dim32(0), C = X.dim32(1);
  CAFFE_ENFORCE(filter.ndim(), 5);
  const int M = filter.dim32(0);

  CAFFE_ENFORCE(filter.dim32(1) == C, "");
  CAFFE_ENFORCE(filter.dim32(2) == this->kernel_h_, "");
  CAFFE_ENFORCE(filter.dim32(3) == this->kernel_w_, "");
  CAFFE_ENFORCE(bias.ndim() == 1, "");
  CAFFE_ENFORCE(bias.dim32(0) == M, "");

  ConvPoolOpBase<MetalCaffeContext>::SetOutputSize(X, Y, filter.dim32(0));

  id<MTLBuffer> inputDataBuffer = GetMetalAllocator()->Buffer((void*)X.template data<float>());
  id<MTLBuffer> outputDataBuffer = GetMetalAllocator()->Buffer((void*)Y->template mutable_data<float>());
  id<MTLBuffer> weightBuffer = GetMetalAllocator()->Buffer((void*)filter.template data<uint16_t>());
  const float* biasData = bias.template data<float>();

  metal_convolution(
      inputDataBuffer,
      X.dim32(1),
      X.dim32(3),
      X.dim32(2),
      stride_h_,
      stride_w_,
      pad_t_,
      pad_l_,
      pad_b_,
      pad_r_,
      weightBuffer,
      filter.dim32(0),
      filter.dim32(1),
      filter.dim32(3),
      filter.dim32(2),
      outputDataBuffer,
      Y->dim32(1),
      Y->dim32(3),
      Y->dim32(2),
      biasData,
      bias.dim32(0),
      false);

  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(MetalConv, METAL, MetalConvMTLBufferOp);
OPERATOR_SCHEMA(MetalConv).NumInputs(3).NumOutputs(1);

class MetalConvTransposeMTLBufferOp final : public ConvTransposeUnpoolBase<MetalCaffeContext> {
 public:
  MetalConvTransposeMTLBufferOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvTransposeUnpoolBase<MetalCaffeContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW, "Metal only supports NCHW order.");
  }

  bool RunOnDeviceWithOrderNCHW() override;

 private:
  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);

  const TensorCPU& InputFromTensorCPU(int idx) {
    const Blob*                           blob = Inputs()[idx];
    return blob->Get<TensorCPU>();
  }
};

bool MetalConvTransposeMTLBufferOp::RunOnDeviceWithOrderNCHW() {
  const TensorMetal& X      = Input(INPUT);
  auto&              filter = Input(FILTER);
  auto&              bias   = InputFromTensorCPU(BIAS);
  TensorMetal*       Y      = Output(0);

  const int N = X.dim32(0), M = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
  CAFFE_ENFORCE(filter.ndim() == 5, "filter must be 4D tensor");
  CAFFE_ENFORCE(filter.dim32(0) == M, "filter number must be equal to input channel number");
  const int C = filter.dim32(1);
  CAFFE_ENFORCE(filter.dim32(2) == kernel_h_, "filter height must be equal to kernel height");
  CAFFE_ENFORCE(filter.dim32(3) == kernel_w_, "filter width must be equal to kernel width");
  CAFFE_ENFORCE(bias.ndim() == 1, "bias must be 1D tensor");
  CAFFE_ENFORCE(bias.dim32(0) == C, "bias dimension must be equal to output channel number");

  ConvTransposeUnpoolBase<MetalCaffeContext>::SetOutputSize(X, Y, C);

  id<MTLBuffer> inputDataBuffer = GetMetalAllocator()->Buffer((void*)X.template data<float>());
  id<MTLBuffer> outputDataBuffer = GetMetalAllocator()->Buffer((void*)Y->template mutable_data<float>());
  id<MTLBuffer> weightBuffer = GetMetalAllocator()->Buffer((void*)filter.template data<uint16_t>());
  const float* biasData = bias.template data<float>();

  metal_convolution(
      inputDataBuffer,
      X.dim32(1),
      X.dim32(3),
      X.dim32(2),
      stride_h_,
      stride_w_,
      pad_t_,
      pad_l_,
      pad_b_,
      pad_r_,
      weightBuffer,
      filter.dim32(0),
      filter.dim32(1),
      filter.dim32(3),
      filter.dim32(2),
      outputDataBuffer,
      Y->dim32(1),
      Y->dim32(3),
      Y->dim32(2),
      biasData,
      bias.dim32(0),
      true);
  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(MetalConvTranspose, METAL, MetalConvTransposeMTLBufferOp);
OPERATOR_SCHEMA(MetalConvTranspose).NumInputs(3).NumOutputs(1);
} // namespace caffe2

#endif // CAFFE2_MOBILE
