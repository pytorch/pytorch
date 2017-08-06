// Copyright 2004-present Facebook. All Rights Reserved.

#include "caffe2/core/common.h"

#ifndef CAFFE2_MOBILE
#error "Caffe2 mobile state not defined"
#endif

#if CAFFE2_MOBILE

#include "caffe2/operators/filler_op.h"
#import "data_conversion.h"
#import "FBMetalCNNConvolution.h"
#import "MetalCaffeContext.h"

namespace caffe2 {
class GivenWeightTensorFillOp final : public FillerOp<MetalCaffeContext> {
 public:
  GivenWeightTensorFillOp(const OperatorDef &operator_def, Workspace *ws)
      : FillerOp<MetalCaffeContext>(operator_def, ws) {
    auto source_values = OperatorBase::template GetRepeatedArgument<float>("values");
    for (float f : source_values) {
      values_.push_back(f);
    }
  }

  bool Fill(Tensor<MetalCaffeContext> *output) override { return true; }

  bool RunOnDevice() override {
    auto *output = Operator<MetalCaffeContext>::Output(0);
    CAFFE_ENFORCE_EQ(shape_.size(), 4);
    auto shape = shape_;

    int kernels = shape[0];
    int kernel_channels = shape[1];
    int kernel_height = shape[2];
    int kernel_width = shape[3];

    reformatKernelImage<weight_buffer_type>(
        (const float *)values_.data(),
        kernels,
        kernel_channels,
        kernel_width,
        kernel_height,
        false,
        [&](int multiplier) -> weight_buffer_type * {
          // buffer_size = kernels * kernel_channels * kernel_width * kernel_height * multiplier;
          shape.push_back(multiplier); // multiplier = ceil(aligned_kernel_stride / kernel_stride)
          output->Resize(shape);

          id<MTLBuffer> weightBuffer = GetMetalAllocator()->Buffer((void *)output->template mutable_data<uint16_t>());
          weight_buffer_type *output_data = (weight_buffer_type *)[weightBuffer contents];
          return output_data;
        });
    return true;
  }

 private:
  vector<float> values_;
};

REGISTER_CPU_OPERATOR_WITH_ENGINE(GivenWeightTensorFill, METAL, GivenWeightTensorFillOp);
OPERATOR_SCHEMA(GivenWeightTensorFill).NumInputs(0, 1).NumOutputs(1).AllowInplace({{0, 0}});

class GivenTransposeWeightTensorFillOp final : public FillerOp<MetalCaffeContext> {
 public:
  GivenTransposeWeightTensorFillOp(const OperatorDef &operator_def, Workspace *ws)
      : FillerOp<MetalCaffeContext>(operator_def, ws) {
    auto source_values = OperatorBase::template GetRepeatedArgument<float>("values");
    for (float f : source_values) {
      values_.push_back(f);
    }
  }

  bool Fill(Tensor<MetalCaffeContext> *output) override { return true; }

  bool RunOnDevice() override {
    auto *output = Operator<MetalCaffeContext>::Output(0);
    CAFFE_ENFORCE_EQ(shape_.size(), 4);
    auto shape = shape_;

    int kernels = shape[0];
    int kernel_channels = shape[1];
    int kernel_height = shape[2];
    int kernel_width = shape[3];

    std::swap(kernels, kernel_channels);

    reformatKernelImage<weight_buffer_type>(
        (const float *)values_.data(),
        kernels,
        kernel_channels,
        kernel_width,
        kernel_height,
        true,
        [&](int multiplier) -> weight_buffer_type * {
          // buffer_size = kernels * kernel_channels * kernel_width * kernel_height * multiplier;
          shape_.push_back(multiplier); // multiplier = ceil(aligned_kernel_stride / kernel_stride)
          output->Resize(shape_);

          id<MTLBuffer> weightBuffer = GetMetalAllocator()->Buffer((void *)output->template mutable_data<uint16_t>());
          weight_buffer_type *output_data = (weight_buffer_type *)[weightBuffer contents];
          return output_data;
        });
    return true;
  }

 private:
  vector<float> values_;
};

REGISTER_CPU_OPERATOR_WITH_ENGINE(GivenTransposeWeightTensorFill, METAL, GivenTransposeWeightTensorFillOp);
OPERATOR_SCHEMA(GivenTransposeWeightTensorFill).NumInputs(0, 1).NumOutputs(1).AllowInplace({{0, 0}});

class CopyWeightTensorToMetalGPUOp final : public Operator<MetalCaffeContext> {
 public:
  CopyWeightTensorToMetalGPUOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<MetalCaffeContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    const Blob *blob = Inputs()[0];
    const TensorCPU &X = blob->Get<TensorCPU>();
    auto *output = Operator<MetalCaffeContext>::Output(0);
    auto shape = X.dims();
    CAFFE_ENFORCE_EQ(shape.size(), 4);

    int kernels = shape[0];
    int kernel_channels = shape[1];
    int kernel_height = shape[2];
    int kernel_width = shape[3];

    reformatKernelImage<weight_buffer_type>(
        (const float *)X.template data<float>(),
        kernels,
        kernel_channels,
        kernel_width,
        kernel_height,
        false,
        [&](int multiplier) -> weight_buffer_type * {
          // buffer_size = kernels * kernel_channels * kernel_width * kernel_height * multiplier;
          shape.push_back(multiplier); // multiplier = ceil(aligned_kernel_stride / kernel_stride)
          output->Resize(shape);

          id<MTLBuffer> weightBuffer = GetMetalAllocator()->Buffer((void *)output->template mutable_data<uint16_t>());
          weight_buffer_type *output_data = (weight_buffer_type *)[weightBuffer contents];
          return output_data;
        });
    return true;
  }
};

REGISTER_CPU_OPERATOR_WITH_ENGINE(CopyWeightTensorToMetalGPU, METAL, CopyWeightTensorToMetalGPUOp);
OPERATOR_SCHEMA(CopyWeightTensorToMetalGPU).NumInputs(0, 1).NumOutputs(1).AllowInplace({{0, 0}});

class CopyTransposeWeightTensorToMetalGPUOp final : public Operator<MetalCaffeContext> {
 public:
  CopyTransposeWeightTensorToMetalGPUOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<MetalCaffeContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    const Blob *blob = Inputs()[0];
    const TensorCPU &X = blob->Get<TensorCPU>();
    auto *output = Operator<MetalCaffeContext>::Output(0);
    auto shape_ = X.dims();
    CAFFE_ENFORCE_EQ(shape_.size(), 4);
    auto shape = shape_;

    int kernels = shape[0];
    int kernel_channels = shape[1];
    int kernel_height = shape[2];
    int kernel_width = shape[3];

    std::swap(kernels, kernel_channels);

    reformatKernelImage<weight_buffer_type>(
        (const float *)X.template data<float>(),
        kernels,
        kernel_channels,
        kernel_width,
        kernel_height,
        true,
        [&](int multiplier) -> weight_buffer_type * {
          // buffer_size = kernels * kernel_channels * kernel_width * kernel_height * multiplier;
          shape_.push_back(multiplier); // multiplier = ceil(aligned_kernel_stride / kernel_stride)
          output->Resize(shape_);

          id<MTLBuffer> weightBuffer = GetMetalAllocator()->Buffer((void *)output->template mutable_data<uint16_t>());
          weight_buffer_type *output_data = (weight_buffer_type *)[weightBuffer contents];
          return output_data;
        });
    return true;
  }
};

REGISTER_CPU_OPERATOR_WITH_ENGINE(CopyTransposeWeightTensorToMetalGPU, METAL, CopyTransposeWeightTensorToMetalGPUOp);
OPERATOR_SCHEMA(CopyTransposeWeightTensorToMetalGPU).NumInputs(0, 1).NumOutputs(1).AllowInplace({{0, 0}});

} // namespace caffe2

#endif // CAFFE2_MOBILE
