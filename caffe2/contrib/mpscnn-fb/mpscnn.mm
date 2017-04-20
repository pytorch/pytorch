// Copyright 2004-present Facebook. All Rights Reserved.

#include "caffe2/core/common.h"
#include "caffe2/core/context.h"

#if CAFFE2_MOBILE

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/conv_transpose_unpool_op_base.h"
#include "caffe2/operators/spatial_batch_norm_op.h"

#include "mpscnn_context.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MPSCNN.h>

#import <UIKit/UIDevice.h>

namespace caffe2 {

namespace {

static constexpr const char* kMPSCNNReadCountArg = "__mpscnn_read_count__";

auto divRoundUp(uint x, uint y) -> uint { return (x + y - 1) / y; }

NSString* kernelFor(const MPSTemporaryImage* X, NSString* arrayKernel, NSString* nonArrayKernel) {
  if (X.featureChannels > 4) {
    return arrayKernel;
  }
  if (X.numberOfImages > 1) {
    return arrayKernel;
  }
  return nonArrayKernel;
}

struct LaunchParams {
  MTLSize threadsPerThreadgroup;
  MTLSize threadgroupsPerGrid;
};

LaunchParams spatialPointwiseKernelLaunchParams(id<MTLComputePipelineState> pipeline,
                                                const MPSTemporaryImage* im) {
  const auto maxThreadsPerThreadgroup = [pipeline maxTotalThreadsPerThreadgroup];
  const auto threadExecutionWidth = [pipeline threadExecutionWidth];
  const auto threadsPerThreadgroup = MTLSizeMake(
      8 /* threadExecutionWidth */, 4 /* maxThreadsPerThreadgroup / threadExecutionWidth */, 1);
  const auto threadgroupsPerGrid =
      MTLSizeMake(divRoundUp(im.width, threadsPerThreadgroup.width),
                  divRoundUp(im.height, threadsPerThreadgroup.height),
                  im.numberOfImages * divRoundUp(im.featureChannels, 4));
  return {threadsPerThreadgroup, threadgroupsPerGrid};
};

void computeOutputHW(ConvPoolOpBase<CPUContext>* op, int H, int W, int* OH, int* OW) {
  Tensor<CPUContext> input, output;
  input.Resize(1, 1, H, W);
  op->SetOutputSize<CPUContext>(input, &output, 1);
  CAFFE_ENFORCE_EQ(output.ndim(), 4);
  *OH = output.dim(2);
  *OW = output.dim(3);
}

constexpr int computeMPSAlignOffset(int kernel, int pad) {
  // To set the offset, we can just match the top-left pixel (in the input image, with negative
  // values for padding)
  // that we look at.
  // For 3x3s1p1, we look at the (-1, -1) pixel in the original impl.
  // For 3x3s1p0, we look at (0, 0) pixel.
  // For 3x3s1p2, look at (-2, -2)
  // MPSCNN always looks at (-floor(kernel_size - 1 / 2), -floor(kernel_size - 1 / 2))
  // Thus, we just need to match this up.

  // For 3x3s1p1, offset should be (0, 0)
  // For 3x3s1p0, offset should be (1, 1)
  // For 3x3s1p2, offset should be (-1, -1)
  const int mps_offset = kernel % 2 == 0 ? kernel / 2 : (kernel - 1) / 2;
  const int c2_offset = pad;
  return mps_offset - c2_offset;
};

MPSTemporaryImage* createImage(const OperatorBase* op,
                               id<MTLCommandBuffer> commandBuffer,
                               int n,
                               int height,
                               int width,
                               int channels,
                               size_t output_idx = 0) {

  auto* image = [MPSTemporaryImage
      temporaryImageWithCommandBuffer:commandBuffer
                      imageDescriptor:
                          [MPSImageDescriptor
                              imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                                         width:width
                                                        height:height
                                               featureChannels:channels
                                                numberOfImages:n
                                                         usage:MTLTextureUsageShaderRead |
                                                               MTLTextureUsageShaderWrite]];
  // We'll try to look at the per-output_idx read-count argument, otherwise, we'll use the
  // operator-global default.
  const auto& readCounts = op->GetRepeatedArgument<int>(kMPSCNNReadCountArg);
  const auto readCount = readCounts.size() ? readCounts.at(output_idx)
                                           : op->GetSingleArgument<int>(kMPSCNNReadCountArg, 1);
  CAFFE_ENFORCE_GE(readCount, 1);
  image.readCount = readCount;
  return image;
}

struct MPSImageWrapper {
  void* x;
  void* commandBuffer_;
};

class CopyToMPSCNNOp final : public Operator<CPUContext> {
 public:
  CopyToMPSCNNOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    commandBuffer_ = [getMPSCNNContext().commandQueue commandBuffer];
    inputBuffers_.resize(Inputs().size());
    output_.resize(Inputs().size());
    for (auto i = 0; i < Inputs().size(); ++i) {
      const auto& X = Input(i);
      CAFFE_ENFORCE_EQ(X.ndim(), 4);
      caffe2::Timer t;
      const auto n = X.dim(0);
      const auto width = X.dim(3);
      const auto height = X.dim(2);
      const auto channels = X.dim(1);

      caffe2::Timer copyT;
      if (!inputBuffers_[i] || inputBuffers_[i].length != X.nbytes()) {
        inputBuffers_[i] = [getMPSCNNContext().device
            newBufferWithLength:X.nbytes()
                        options:MTLResourceOptionCPUCacheModeWriteCombined];
      }
      memcpy([inputBuffers_[i] contents], X.raw_data(), X.nbytes());
      VLOG(2) << "CopyToMPSCNNOp input copy took: " << copyT.MilliSeconds();

      output_[i] = createImage(this, commandBuffer_, n, height, width, channels, i);
      id<MTLComputeCommandEncoder> encoder = [commandBuffer_ computeCommandEncoder];
      id<MTLComputePipelineState> state = getMPSCNNContext().getSpecializedPipelineState(
          kernelFor(output_[i], @"copy_nchw_to_metal", @"copy_nchw_to_metal_nonarray"),
          {{ushort(channels), ushort(height), ushort(width)}});
      [encoder setComputePipelineState:state];
      [encoder setBuffer:inputBuffers_[i] offset:0 atIndex:0];
      [encoder setTexture:[output_[i] texture] atIndex:0];
      const auto& launchParams = spatialPointwiseKernelLaunchParams(state, output_[i]);
      [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
              threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
      [encoder endEncoding];
      VLOG(2) << "CopyToMPSCNNOp took: " << t.MilliSeconds();
      Outputs()[i]->GetMutable<MPSImageWrapper>()->x = (__bridge void*)output_[i];
      Outputs()[i]->GetMutable<MPSImageWrapper>()->commandBuffer_ = (__bridge void*)commandBuffer_;
    }

    return true;
  }

 private:
  std::vector<id<MTLBuffer>> inputBuffers_;
  id<MTLCommandBuffer> commandBuffer_{nullptr};
  std::vector<MPSTemporaryImage*> output_{nullptr};
};

REGISTER_CPU_OPERATOR(CopyToMPSCNN, CopyToMPSCNNOp);
OPERATOR_SCHEMA(CopyToMPSCNN).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX).SameNumberOfOutput();

auto mpsImageSize = [](MPSTemporaryImage* X) {
  return X.featureChannels * X.height * X.width * X.numberOfImages;
};

class CopyFromMPSCNNOp final : public Operator<CPUContext> {
 public:
  CopyFromMPSCNNOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    caffe2::Timer t;
    auto cb = [&](size_t i) { return Inputs()[i]->template Get<MPSImageWrapper>().commandBuffer_; };
    auto X = [&](size_t i) {
      return (__bridge MPSTemporaryImage*)Inputs()[i]->template Get<MPSImageWrapper>().x;
    };
    auto cb0 = (__bridge id<MTLCommandBuffer>)cb(0);
    outputBuffers_.resize(Inputs().size());
    for (auto i = 0; i < Inputs().size(); ++i) {
      CAFFE_ENFORCE_EQ(cb0, cb(i));
      MPSTemporaryImage* Xi = X(i);
      if (!outputBuffers_[i] || outputBuffers_[i].length != mpsImageSize(Xi) * sizeof(float)) {
        outputBuffers_[i] =
            [getMPSCNNContext().device newBufferWithLength:mpsImageSize(Xi) * sizeof(float)
                                                   options:MTLResourceOptionCPUCacheModeDefault];
      }
      id<MTLComputeCommandEncoder> encoder = [cb0 computeCommandEncoder];
      id<MTLComputePipelineState> state = getMPSCNNContext().getSpecializedPipelineState(
          kernelFor(Xi, @"copy_metal_to_nchw", @"copy_metal_to_nchw_nonarray"),
          {{ushort(Xi.featureChannels), ushort(Xi.height), ushort(Xi.width)}});

      [encoder setComputePipelineState:state];
      [encoder setBuffer:outputBuffers_[i] offset:0 atIndex:0];
      [encoder setTexture:[Xi texture] atIndex:0];

      const auto& launchParams = spatialPointwiseKernelLaunchParams(state, Xi);
      [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
              threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
      [encoder endEncoding];
      Xi.readCount -= 1;
    }

    dispatch_semaphore_t gpu_execution_done = dispatch_semaphore_create(0);
    [cb0 addCompletedHandler:^(id<MTLCommandBuffer> currentBuffer) {
      if ([currentBuffer error] != nil) {
        LOG(ERROR) << "Metal execution failed: "
                   << [[[currentBuffer error] localizedDescription] UTF8String];
      }
      dispatch_semaphore_signal(gpu_execution_done);
    }];
    [cb0 commit];

    {
      caffe2::Timer wt;
      dispatch_semaphore_wait(gpu_execution_done, DISPATCH_TIME_FOREVER);
      VLOG(2) << "CopyFromMPSCNNOp semaphore wait took: " << wt.MilliSeconds();
    }
    for (auto i = 0; i < Inputs().size(); ++i) {
      caffe2::Timer copyOutT;
      MPSTemporaryImage* Xi = X(i);
      Output(i)->Resize(Xi.numberOfImages, Xi.featureChannels, Xi.height, Xi.width);
      Output(i)->mutable_data<float>();
      CAFFE_ENFORCE_EQ(outputBuffers_[i].length, Output(i)->nbytes());
      memcpy(
          Output(i)->mutable_data<float>(), [outputBuffers_[i] contents], outputBuffers_[i].length);
      VLOG(2) << "CopyFromMPSCNNOp memcpy took: " << copyOutT.MilliSeconds();
    }
    VLOG(2) << "CopyFromMPSCNNOp took: " << t.MilliSeconds();
    return true;
  }

 private:
  std::vector<id<MTLBuffer>> outputBuffers_;
};

REGISTER_CPU_OPERATOR(CopyFromMPSCNN, CopyFromMPSCNNOp);
OPERATOR_SCHEMA(CopyFromMPSCNN).NumInputs(1, INT_MAX).NumOutputs(1, INT_MAX).SameNumberOfOutput();

class MPSCNNPackedInt8BGRANHWCToNCHWCStylizerPreprocessOp final : public Operator<CPUContext> {
 public:
  MPSCNNPackedInt8BGRANHWCToNCHWCStylizerPreprocessOp(const OperatorDef& operator_def,
                                                      Workspace* ws)
      : Operator<CPUContext>(operator_def, ws), ws_(ws) {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    const auto& mean = Input(1);
    CAFFE_ENFORCE_EQ(mean.size(), 3);
    CAFFE_ENFORCE_EQ(X.ndim(), 4);
    CAFFE_ENFORCE_EQ(X.dim(0), 1);
    CAFFE_ENFORCE_EQ(X.dim(3), 4);
    const auto H = X.dim(1);
    const auto W = X.dim(2);

    caffe2::Timer t;

    auto* noiseBlob = ws_->CreateBlob("__CAFFE2_STYLIZER_NOISE__");
    ushort noiseSize =
        OperatorBase::GetSingleArgument<int>("noise_size", 491 /* prime to avoid artifacts */);
    // Treaded as half4 in the kernel, so need half4 here.
    noiseSize = divRoundUp(noiseSize, 4) * 4;
    if (!noiseBlob->IsType<TensorCPU>() || noiseBlob->Get<TensorCPU>().size() != noiseSize) {
      VLOG(2) << "Initializing stylizer with noise: " << noiseSize;
      caffe2::Timer rt;
      // Initialize random noise on first use.
      // Cache it to maintain temporal consistency.
      auto* t = noiseBlob->template GetMutable<TensorCPU>();
      t->Resize(noiseSize);
      math::RandGaussian<float, CPUContext>(
          t->size(),
          0.0,
          OperatorBase::GetSingleArgument<float>("noise_std", 10.0),
          t->template mutable_data<float>(),
          &context_);
      VLOG(2) << "Preprocess initializing noise: " << rt.MilliSeconds();
    }
    const auto& noise = noiseBlob->Get<TensorCPU>();

    if (!inputBuffer_ || inputBuffer_.length != X.nbytes()) {
      caffe2::Timer pt;

      inputBuffer_ = [getMPSCNNContext().device
          newBufferWithLength:X.nbytes()
                      options:MTLResourceOptionCPUCacheModeWriteCombined];
      meanBuffer_ = [getMPSCNNContext().device
          newBufferWithLength:4 * 2 // (3/4 half-floats).
                      options:MTLResourceOptionCPUCacheModeWriteCombined];
      noiseBuffer_ = [getMPSCNNContext().device
          newBufferWithLength:noiseSize * sizeof(float16_t)
                      options:MTLResourceOptionCPUCacheModeWriteCombined];

      float16_t* meanBufferPtr = (float16_t*)[meanBuffer_ contents];
      CAFFE_ENFORCE(meanBufferPtr);
      for (auto i = 0; i < mean.size(); ++i) {
        meanBufferPtr[i] = mean.data<float>()[i];
      }
      float16_t* noiseBufferPtr = (float16_t*)[noiseBuffer_ contents];
      CAFFE_ENFORCE(noiseBufferPtr);
      for (auto i = 0; i < mean.size(); ++i) {
        noiseBufferPtr[i] = noise.data<float>()[i];
      }

      VLOG(2) << "Preprocess construct took: " << pt.MilliSeconds();
    }

    {
      caffe2::Timer ct;
      memcpy([inputBuffer_ contents], X.raw_data(), X.nbytes());
      VLOG(2) << "Preprocess memcpy took: " << ct.MilliSeconds();
    }
    commandBuffer_ = [getMPSCNNContext().commandQueue commandBuffer];
    output_ = createImage(this, commandBuffer_, 1, H, W, 3);

    id<MTLComputeCommandEncoder> encoder = [commandBuffer_ computeCommandEncoder];
    id<MTLComputePipelineState> state =
        getMPSCNNContext().getSpecializedPipelineState(@"preprocess_stylizer", {noiseSize});

    [encoder setComputePipelineState:state];
    [encoder setBuffer:inputBuffer_ offset:0 atIndex:0];
    [encoder setBuffer:meanBuffer_ offset:0 atIndex:1];
    [encoder setBuffer:noiseBuffer_ offset:0 atIndex:2];

    [encoder setTexture:[output_ texture] atIndex:0];
    const auto& launchParams = spatialPointwiseKernelLaunchParams(state, output_);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    Outputs()[0]->GetMutable<MPSImageWrapper>()->x = (__bridge void*)output_;
    Outputs()[0]->GetMutable<MPSImageWrapper>()->commandBuffer_ = (__bridge void*)commandBuffer_;

    VLOG(2) << "Preprocess took: " << t.MilliSeconds();
    return true;
  }

 private:
  Workspace* ws_{nullptr};
  id<MTLBuffer> inputBuffer_{nullptr};
  id<MTLCommandBuffer> commandBuffer_{nullptr};
  id<MTLBuffer> noiseBuffer_{nullptr};
  id<MTLBuffer> meanBuffer_{nullptr};
  MPSTemporaryImage* output_{nullptr};
};

REGISTER_CPU_OPERATOR(MPSCNNPackedInt8BGRANHWCToNCHWCStylizerPreprocess,
                      MPSCNNPackedInt8BGRANHWCToNCHWCStylizerPreprocessOp);
OPERATOR_SCHEMA(MPSCNNPackedInt8BGRANHWCToNCHWCStylizerPreprocess).NumInputs(2).NumOutputs(1);

class MPSCNNBRGNCHWCToPackedInt8BGRAStylizerDeprocessOp final : public Operator<CPUContext> {
 public:
  MPSCNNBRGNCHWCToPackedInt8BGRAStylizerDeprocessOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    MPSTemporaryImage* X = (__bridge MPSTemporaryImage*)(Inputs()[0]->Get<MPSImageWrapper>().x);
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)(Inputs()[0]->Get<MPSImageWrapper>().commandBuffer_);

    const auto& mean = Input(1);
    caffe2::Timer t;
    const auto W = X.width;
    const auto H = X.height;
    CAFFE_ENFORCE_EQ(X.featureChannels, 3);
    CAFFE_ENFORCE_EQ(X.numberOfImages, 1);

    if (!outputBuffer_ || outputBuffer_.length != X.height * X.width * 4) {
      caffe2::Timer pt;

      outputBuffer_ =
          [getMPSCNNContext().device newBufferWithLength:X.height * X.width * 4
                                                 options:MTLResourceOptionCPUCacheModeDefault];
      meanBuffer_ = [getMPSCNNContext().device
          newBufferWithLength:4 * 2 // (3/4 half-floats).
                      options:MTLResourceOptionCPUCacheModeWriteCombined];
      float16_t* meanBufferPtr = (float16_t*)[meanBuffer_ contents];
      for (auto i = 0; i < mean.size(); ++i) {
        meanBufferPtr[i] = mean.data<float>()[i];
      }
      VLOG(2) << "Deprocess copy took: " << pt.MilliSeconds();
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state = getMPSCNNContext().getPipelineState(@"deprocess_stylizer");

    CAFFE_ENFORCE_EQ(outputBuffer_.length, X.height * X.width * 4);
    [encoder setComputePipelineState:state];
    [encoder setBuffer:outputBuffer_ offset:0 atIndex:0];
    [encoder setBuffer:meanBuffer_ offset:0 atIndex:1];
    [encoder setTexture:[X texture] atIndex:0];
    const auto& launchParams = spatialPointwiseKernelLaunchParams(state, X);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    X.readCount -= 1;
    dispatch_semaphore_t gpu_execution_done = dispatch_semaphore_create(0);
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> currentBuffer) {
      if ([currentBuffer error] != nil) {
        LOG(ERROR) << "Metal execution failed: "
                   << [[[currentBuffer error] localizedDescription] UTF8String];
      }
      dispatch_semaphore_signal(gpu_execution_done);
    }];
    [commandBuffer commit];

    {
      caffe2::Timer wt;
      dispatch_semaphore_wait(gpu_execution_done, DISPATCH_TIME_FOREVER);
      VLOG(2) << "Deprocess semaphore wait took: " << wt.MilliSeconds();
    }
    Output(0)->Resize(1, X.height, X.width, 4);
    {
      caffe2::Timer ct;
      memcpy(Output(0)->mutable_data<uint8_t>(), [outputBuffer_ contents], [outputBuffer_ length]);
      VLOG(2) << "Deprocess copy: " << t.MilliSeconds();
    }
    CAFFE_ENFORCE_EQ(Output(0)->nbytes(), [outputBuffer_ length]);
    VLOG(2) << "Deprocess took: " << t.MilliSeconds();

    return true;
  }

 private:
  id<MTLBuffer> outputBuffer_{nullptr};
  id<MTLBuffer> meanBuffer_{nullptr};
};

REGISTER_CPU_OPERATOR(MPSCNNBRGNCHWCToPackedInt8BGRAStylizerDeprocess,
                      MPSCNNBRGNCHWCToPackedInt8BGRAStylizerDeprocessOp);
OPERATOR_SCHEMA(MPSCNNBRGNCHWCToPackedInt8BGRAStylizerDeprocess).NumInputs(2).NumOutputs(1);

template <typename Neuron>
class MPSCNNNeuronOp final : public Operator<CPUContext> {
 public:
  MPSCNNNeuronOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    caffe2::Timer t;

    MPSTemporaryImage* X =
        (__bridge MPSTemporaryImage*)(Inputs()[0]->template Get<MPSImageWrapper>().x);
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)(Inputs()[0]
                                            ->template Get<MPSImageWrapper>()
                                            .commandBuffer_);

    output_ =
        createImage(this, commandBuffer, X.numberOfImages, X.height, X.width, X.featureChannels);
    CAFFE_ENFORCE_EQ(output_.width, X.width);
    CAFFE_ENFORCE_EQ(output_.height, X.height);
    CAFFE_ENFORCE_EQ(output_.featureChannels, X.featureChannels);

    if (!neuron_) {
      neuron_ = Neuron::t();
    }
    [neuron_ encodeToCommandBuffer:commandBuffer sourceImage:X destinationImage:output_];
    Outputs()[0]->template GetMutable<MPSImageWrapper>()->x = (__bridge void*)output_;
    Outputs()[0]->template GetMutable<MPSImageWrapper>()->commandBuffer_ =
        (__bridge void*)commandBuffer;

    VLOG(2) << "ElementwiseAdd took: " << t.MilliSeconds();
    return true;
  }
  MPSCNNNeuron* neuron_{nullptr};
  MPSTemporaryImage* output_{nullptr};
};

#define INIT_NEURON_OP(n)                                          \
  REGISTER_CPU_OPERATOR(MPSCNN##n, MPSCNNNeuronOp<n##NeuronInit>); \
  OPERATOR_SCHEMA(MPSCNN##n).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});

struct SigmoidNeuronInit {
  static MPSCNNNeuron* t() {
    return [[MPSCNNNeuronSigmoid alloc] initWithDevice:getMPSCNNContext().device];
  }
};
INIT_NEURON_OP(Sigmoid);

struct ReluNeuronInit {
  static MPSCNNNeuron* t() {
    return [[MPSCNNNeuronReLU alloc] initWithDevice:getMPSCNNContext().device a:0];
  }
};
INIT_NEURON_OP(Relu);

struct TanhNeuronInit {
  static MPSCNNNeuron* t() {
    return [[MPSCNNNeuronTanH alloc] initWithDevice:getMPSCNNContext().device a:1 b:1];
  }
};
INIT_NEURON_OP(Tanh);

#undef INIT_NEURON_OP

template <typename Neuron>
class MPSCNNConvOp final : public ConvPoolOpBase<CPUContext> {
 public:
  MPSCNNConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW, "Metal only supports NCHW order.");
  }

  bool RunOnDeviceWithOrderNCHW() override {
    caffe2::Timer t;

    MPSTemporaryImage* X =
        (__bridge MPSTemporaryImage*)(Inputs()[0]->template Get<MPSImageWrapper>().x);
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)(Inputs()[0]
                                            ->template Get<MPSImageWrapper>()
                                            .commandBuffer_);

    auto& filter = Input(FILTER);
    auto& bias = Input(BIAS);
    CAFFE_ENFORCE(filter.ndim(), 4);
    const int C = filter.dim32(1);
    const int M = filter.dim32(0);

    CAFFE_ENFORCE(X.featureChannels == C, "");
    CAFFE_ENFORCE(filter.dim32(2) == this->kernel_h_, "");
    CAFFE_ENFORCE(filter.dim32(3) == this->kernel_w_, "");
    CAFFE_ENFORCE(bias.ndim() == 1, "");
    CAFFE_ENFORCE(bias.dim32(0) == M, "");

    const auto kH = this->kernel_h_;
    const auto kW = this->kernel_w_;

    // ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));
    // Reformat weights from [M][C][kH][kW] to [M][kH][kW][C].
    if (!conv_) {
      caffe2::Timer consT;
      std::vector<float> refilter(M * kH * kW * C);
      auto* filter_ = filter.template data<float>();
      for (auto m = 0; m < M; ++m) {
        for (auto c = 0; c < C; ++c) {
          for (auto kh = 0; kh < kH; ++kh) {
            for (auto kw = 0; kw < kW; ++kw) {
              // refilter[m][kh][kw][c]
              refilter[m * kH * kW * C + kh * kW * C + kw * C + c] =
                  // filter[m][c][kh][kw]
                  filter_[m * C * kH * kW + c * kH * kW + kh * kW + kw];
            }
          }
        }
      }

      MPSCNNConvolutionDescriptor* desc =
          [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:kW
                                                                  kernelHeight:kH
                                                          inputFeatureChannels:C
                                                         outputFeatureChannels:M
                                                                  neuronFilter:Neuron::t()];
      desc.strideInPixelsX = this->stride_w_;
      desc.strideInPixelsY = this->stride_h_;

      conv_ = [[MPSCNNConvolution alloc] initWithDevice:getMPSCNNContext().device
                                  convolutionDescriptor:desc
                                          kernelWeights:refilter.data()
                                              biasTerms:bias.template data<float>()
                                                  flags:MPSCNNConvolutionFlagsNone];
      [conv_ setEdgeMode:MPSImageEdgeModeZero];

      MPSOffset offset;
      offset.x = computeMPSAlignOffset(kW, pad_l_);
      offset.y = computeMPSAlignOffset(kH, pad_t_);
      offset.z = 0;
      [conv_ setOffset:offset];
      VLOG(2) << "MPSCNNConv ConvDesc took: " << consT.MilliSeconds();
    }

    CAFFE_ENFORCE_EQ(conv_.strideInPixelsY, this->stride_h_);
    CAFFE_ENFORCE_EQ(conv_.strideInPixelsX, this->stride_w_);
    CAFFE_ENFORCE_EQ(conv_.groups, 1);
    CAFFE_ENFORCE_EQ(conv_.inputFeatureChannels, C);
    CAFFE_ENFORCE_EQ(conv_.outputFeatureChannels, M);
    CAFFE_ENFORCE_EQ(conv_.kernelWidth, kW);
    CAFFE_ENFORCE_EQ(conv_.kernelHeight, kH);

    int output_height;
    int output_width;
    computeOutputHW(this, X.height, X.width, &output_height, &output_width);
    int output_channels = M;

    VLOG(2) << "Output height: " << output_height;
    VLOG(2) << "Output width:" << output_width;
    VLOG(2) << "Output channels:" << output_channels;
    output_ = createImage(
        this, commandBuffer, X.numberOfImages, output_height, output_width, output_channels);
    CAFFE_ENFORCE_EQ(output_.height, output_height);
    CAFFE_ENFORCE_EQ(output_.width, output_width);
    [conv_ encodeToCommandBuffer:commandBuffer sourceImage:X destinationImage:output_];
    Outputs()[0]->template GetMutable<MPSImageWrapper>()->x = (__bridge void*)output_;
    Outputs()[0]->template GetMutable<MPSImageWrapper>()->commandBuffer_ =
        (__bridge void*)commandBuffer;

    VLOG(2) << "MPSCNNConv took: " << t.MilliSeconds();
    return true;
  }

  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);

  MPSCNNConvolution* conv_{nullptr};
  MPSTemporaryImage* output_{nullptr};
};

// No-op init
struct EmptyNeuronInit {
  static MPSCNNNeuron* t() { return nil; }
};
#define INIT_CONV_NEURON_OP(name, neuron)            \
  REGISTER_CPU_OPERATOR(name, MPSCNNConvOp<neuron>); \
  OPERATOR_SCHEMA(name).NumInputs(3).NumOutputs(1);

INIT_CONV_NEURON_OP(MPSCNNConv, EmptyNeuronInit);
INIT_CONV_NEURON_OP(MPSCNNConvRelu, ReluNeuronInit);
INIT_CONV_NEURON_OP(MPSCNNConvSigmoid, SigmoidNeuronInit);

#undef INIT_CONV_NEURON_OP

class MPSCNNPadImageOp final : public ConvPoolOpBase<CPUContext> {
 public:
  MPSCNNPadImageOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW, "Metal only supports NCHW order.");

    OPERATOR_NEEDS_FEATURE(OperatorBase::GetSingleArgument<string>("mode", "") == "reflect",
                           "Metal only supports reflection");
    kernel_h_ = kernel_w_ = 1;
  }

  bool RunOnDeviceWithOrderNCHW() override {
    caffe2::Timer t;

    MPSTemporaryImage* X = (__bridge MPSTemporaryImage*)(Inputs()[0]->Get<MPSImageWrapper>().x);
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)(Inputs()[0]
                                            ->template Get<MPSImageWrapper>()
                                            .commandBuffer_);

    const auto pH = this->pad_t_;
    const auto pW = this->pad_l_;
    const auto output_height = X.height + 2 * pH;
    const auto output_width = X.width + 2 * pW;
    VLOG(1) << "Output height: " << output_height;
    VLOG(1) << "Output width:" << output_width;
    VLOG(2) << "Output channels:" << X.featureChannels;
    output_ = createImage(
        this, commandBuffer, X.numberOfImages, output_height, output_width, X.featureChannels);

    CAFFE_ENFORCE_EQ(output_.height, output_height);
    CAFFE_ENFORCE_EQ(output_.width, output_width);
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state = getMPSCNNContext().getPipelineState(
        kernelFor(output_, @"reflection_padding", @"reflection_padding_nonarray"));
    [encoder setComputePipelineState:state];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[output_ texture] atIndex:1];
    const auto& launchParams = spatialPointwiseKernelLaunchParams(state, output_);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    X.readCount -= 1;
    Outputs()[0]->GetMutable<MPSImageWrapper>()->x = (__bridge void*)output_;
    Outputs()[0]->template GetMutable<MPSImageWrapper>()->commandBuffer_ =
        (__bridge void*)commandBuffer;

    VLOG(2) << "PadImage took: " << t.MilliSeconds();
    return true;
  }

  MPSTemporaryImage* output_{nullptr};
};

REGISTER_CPU_OPERATOR(MPSCNNPadImage, MPSCNNPadImageOp);
OPERATOR_SCHEMA(MPSCNNPadImage).NumInputs(1).NumOutputs(1);

class MPSCNNAddOp final : public Operator<CPUContext> {
 public:
  MPSCNNAddOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    caffe2::Timer t;

    CAFFE_ENFORCE_EQ(Inputs()[1]->template Get<MPSImageWrapper>().commandBuffer_,
                     Inputs()[0]->template Get<MPSImageWrapper>().commandBuffer_);

    MPSTemporaryImage* X0 = (__bridge MPSTemporaryImage*)(Inputs()[0]->Get<MPSImageWrapper>().x);
    MPSTemporaryImage* X1 = (__bridge MPSTemporaryImage*)(Inputs()[1]->Get<MPSImageWrapper>().x);

    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)(Inputs()[0]
                                            ->template Get<MPSImageWrapper>()
                                            .commandBuffer_);
    output_ = createImage(
        this, commandBuffer, X0.numberOfImages, X0.height, X0.width, X0.featureChannels);
    CAFFE_ENFORCE_EQ(X1.width, X0.width);
    CAFFE_ENFORCE_EQ(X1.height, X0.height);
    CAFFE_ENFORCE_EQ(X1.featureChannels, X0.featureChannels);
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state = getMPSCNNContext().getPipelineState(
        kernelFor(X0, @"elementwise_add", @"elementwise_add_nonarray"));

    [encoder setComputePipelineState:state];
    [encoder setTexture:[X0 texture] atIndex:0];
    [encoder setTexture:[X1 texture] atIndex:1];
    [encoder setTexture:[output_ texture] atIndex:2];
    const auto& launchParams = spatialPointwiseKernelLaunchParams(state, output_);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    X0.readCount -= 1;
    X1.readCount -= 1;
    Outputs()[0]->GetMutable<MPSImageWrapper>()->x = (__bridge void*)output_;
    Outputs()[0]->template GetMutable<MPSImageWrapper>()->commandBuffer_ =
        (__bridge void*)commandBuffer;

    VLOG(2) << "ElementwiseAdd took: " << t.MilliSeconds();
    return true;
  }

  MPSTemporaryImage* output_{nullptr};
};

REGISTER_CPU_OPERATOR(MPSCNNAdd, MPSCNNAddOp);
// Not really in-place per-se, but semantically is valid and preserves compatibility.
OPERATOR_SCHEMA(MPSCNNAdd).NumInputs(2).NumOutputs(1).AllowInplace({{0, 0}});

class MPSCNNAveragePoolOp final : public ConvPoolOpBase<CPUContext> {
 public:
  MPSCNNAveragePoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW, "Metal only supports NCHW order.");
  }

  bool RunOnDeviceWithOrderNCHW() override {
    caffe2::Timer t;

    MPSTemporaryImage* X = (__bridge MPSTemporaryImage*)(Inputs()[0]->Get<MPSImageWrapper>().x);
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)(Inputs()[0]
                                            ->template Get<MPSImageWrapper>()
                                            .commandBuffer_);

    if (!pool_ || this->global_pooling_) {
      caffe2::Timer consT;
      this->ComputePads(X.height, X.width);
      pool_ = [[MPSCNNPoolingAverage alloc] initWithDevice:getMPSCNNContext().device
                                               kernelWidth:this->kernel_w_
                                              kernelHeight:this->kernel_h_
                                           strideInPixelsX:this->stride_w_
                                           strideInPixelsY:this->stride_h_];

      [pool_ setEdgeMode:MPSImageEdgeModeClamp];
      MPSOffset offset;
      offset.x = computeMPSAlignOffset(this->kernel_w_, pad_l_);
      offset.y = computeMPSAlignOffset(this->kernel_h_, pad_t_);
      offset.z = 0;
      [pool_ setOffset:offset];
      VLOG(2) << "MPSCNNAveragePool PoolDesc took: " << consT.MilliSeconds();
    }

    CAFFE_ENFORCE_EQ(pool_.strideInPixelsY, this->stride_h_);
    CAFFE_ENFORCE_EQ(pool_.strideInPixelsX, this->stride_w_);
    int output_height;
    int output_width;
    computeOutputHW(this, X.height, X.width, &output_height, &output_width);

    VLOG(2) << "Output height: " << output_height;
    VLOG(2) << "Output width:" << output_width;
    VLOG(2) << "Output channels:" << X.featureChannels;
    output_ = createImage(
        this, commandBuffer, X.numberOfImages, output_height, output_width, X.featureChannels);
    CAFFE_ENFORCE_EQ(output_.height, output_height);
    CAFFE_ENFORCE_EQ(output_.width, output_width);
    [pool_ encodeToCommandBuffer:commandBuffer sourceImage:X destinationImage:output_];
    Outputs()[0]->GetMutable<MPSImageWrapper>()->x = (__bridge void*)output_;
    Outputs()[0]->template GetMutable<MPSImageWrapper>()->commandBuffer_ =
        (__bridge void*)commandBuffer;

    VLOG(2) << "MPSCNNAveragePool took: " << t.MilliSeconds();
    return true;
  }

  MPSCNNPoolingAverage* pool_{nullptr};
  MPSTemporaryImage* output_{nullptr};
};

REGISTER_CPU_OPERATOR(MPSCNNAveragePool, MPSCNNAveragePoolOp);
OPERATOR_SCHEMA(MPSCNNAveragePool).NumInputs(1).NumOutputs(1);

class MPSCNNMaxPoolOp final : public ConvPoolOpBase<CPUContext> {
 public:
  MPSCNNMaxPoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW, "Metal only supports NCHW order.");
  }

  bool RunOnDeviceWithOrderNCHW() override {
    caffe2::Timer t;

    MPSTemporaryImage* X = (__bridge MPSTemporaryImage*)(Inputs()[0]->Get<MPSImageWrapper>().x);
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)(Inputs()[0]
                                            ->template Get<MPSImageWrapper>()
                                            .commandBuffer_);
    if (!pool_ || this->global_pooling_) {
      caffe2::Timer consT;
      this->ComputePads(X.height, X.width);
      pool_ = [[MPSCNNPoolingMax alloc] initWithDevice:getMPSCNNContext().device
                                           kernelWidth:this->kernel_w_
                                          kernelHeight:this->kernel_h_
                                       strideInPixelsX:this->stride_w_
                                       strideInPixelsY:this->stride_h_];

      [pool_ setEdgeMode:MPSImageEdgeModeClamp];
      MPSOffset offset;
      offset.x = computeMPSAlignOffset(this->kernel_w_, pad_l_);
      offset.y = computeMPSAlignOffset(this->kernel_h_, pad_t_);
      offset.z = 0;
      [pool_ setOffset:offset];
      VLOG(2) << "MPSCNNMaxPool PoolDesc took: " << consT.MilliSeconds();
    }

    CAFFE_ENFORCE_EQ(pool_.strideInPixelsY, this->stride_h_);
    CAFFE_ENFORCE_EQ(pool_.strideInPixelsX, this->stride_w_);

    int output_height;
    int output_width;
    computeOutputHW(this, X.height, X.width, &output_height, &output_width);

    VLOG(2) << "Output height: " << output_height;
    VLOG(2) << "Output width:" << output_width;
    VLOG(2) << "Output channels:" << X.featureChannels;
    output_ = createImage(
        this, commandBuffer, X.numberOfImages, output_height, output_width, X.featureChannels);
    CAFFE_ENFORCE_EQ(output_.height, output_height);
    CAFFE_ENFORCE_EQ(output_.width, output_width);
    [pool_ encodeToCommandBuffer:commandBuffer sourceImage:X destinationImage:output_];
    Outputs()[0]->GetMutable<MPSImageWrapper>()->x = (__bridge void*)output_;
    Outputs()[0]->template GetMutable<MPSImageWrapper>()->commandBuffer_ =
        (__bridge void*)commandBuffer;

    VLOG(2) << "MPSCNNMaxPool took: " << t.MilliSeconds();
    return true;
  }

  MPSCNNPoolingMax* pool_{nullptr};
  MPSTemporaryImage* output_{nullptr};
};

REGISTER_CPU_OPERATOR(MPSCNNMaxPool, MPSCNNMaxPoolOp);
OPERATOR_SCHEMA(MPSCNNMaxPool).NumInputs(1).NumOutputs(1);

class MPSCNNSoftmaxOp final : public Operator<CPUContext> {
 public:
  MPSCNNSoftmaxOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    caffe2::Timer t;
    MPSTemporaryImage* X = (__bridge MPSTemporaryImage*)(Inputs()[0]->Get<MPSImageWrapper>().x);
    CAFFE_ENFORCE_EQ(X.height, 1);
    CAFFE_ENFORCE_EQ(X.width, 1);
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)(Inputs()[0]
                                            ->template Get<MPSImageWrapper>()
                                            .commandBuffer_);
    if (!softmax_) {
      softmax_ = [[MPSCNNSoftMax alloc] initWithDevice:getMPSCNNContext().device];
    }
    output_ =
        createImage(this, commandBuffer, X.numberOfImages, X.height, X.width, X.featureChannels);
    [softmax_ encodeToCommandBuffer:commandBuffer sourceImage:X destinationImage:output_];
    Outputs()[0]->GetMutable<MPSImageWrapper>()->x = (__bridge void*)output_;
    Outputs()[0]->template GetMutable<MPSImageWrapper>()->commandBuffer_ =
        (__bridge void*)commandBuffer;
    VLOG(2) << "MPSCNNSoftmax took: " << t.MilliSeconds();
    return true;
  }

  MPSCNNSoftMax* softmax_{nullptr};
  MPSTemporaryImage* output_{nullptr};
};

REGISTER_CPU_OPERATOR(MPSCNNSoftmax, MPSCNNSoftmaxOp);
OPERATOR_SCHEMA(MPSCNNSoftmax).NumInputs(1).NumOutputs(1);

template <typename Neuron>
class MPSCNNFullyConnectedOp final : public Operator<CPUContext> {
 public:
  MPSCNNFullyConnectedOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    caffe2::Timer t;
    MPSTemporaryImage* X =
        (__bridge MPSTemporaryImage*)(Inputs()[0]->template Get<MPSImageWrapper>().x);
    const auto& W = Input(1);
    const auto& b = Input(2);
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)(Inputs()[0]
                                            ->template Get<MPSImageWrapper>()
                                            .commandBuffer_);
    const auto input_channels = W.dim32(1) / X.width / X.height;
    CAFFE_ENFORCE_EQ(input_channels, X.featureChannels);
    const auto output_channels = W.dim32(0);
    if (!fc_) {
      const auto M = output_channels;
      const auto kH = X.height;
      const auto kW = X.width;
      const auto C = input_channels;
      std::vector<float> refilter(M * kH * kW * C);
      auto* filter_ = W.template data<float>();
      for (auto m = 0; m < M; ++m) {
        for (auto c = 0; c < C; ++c) {
          for (auto kh = 0; kh < kH; ++kh) {
            for (auto kw = 0; kw < kW; ++kw) {
              // refilter[m][kh][kw][c]
              refilter[m * kH * kW * C + kh * kW * C + kw * C + c] =
                  // filter[m][c][kh][kw]
                  filter_[m * C * kH * kW + c * kH * kW + kh * kW + kw];
            }
          }
        }
      }

      MPSCNNConvolutionDescriptor* desc =
          [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:X.width
                                                                  kernelHeight:X.height
                                                          inputFeatureChannels:input_channels
                                                         outputFeatureChannels:output_channels
                                                                  neuronFilter:Neuron::t()];
      fc_ = [[MPSCNNConvolution alloc] initWithDevice:getMPSCNNContext().device
                                convolutionDescriptor:desc
                                        kernelWeights:refilter.data()
                                            biasTerms:b.template data<float>()
                                                flags:MPSCNNConvolutionFlagsNone];
    }
    // Note that X.numberOfImages can change between calls, but X.height and X.width are static by
    // definition.
    LOG(INFO) << "MPSCNNFC: " << X.numberOfImages << ", " << X.width << ", " << X.height << ", "
              << X.featureChannels << ", " << output_channels;

    [fc_ setClipRect:MTLRegionMake3D(0, 0, 0, 1, 1, X.numberOfImages)];
    MPSOffset off;
    off.x = X.width / 2;
    off.y = X.height / 2;
    off.z = 0;
    [fc_ setOffset:off];
    output_ = createImage(this, commandBuffer, X.numberOfImages, 1, 1, output_channels);
    [fc_ encodeToCommandBuffer:commandBuffer sourceImage:X destinationImage:output_];
    Outputs()[0]->template GetMutable<MPSImageWrapper>()->x = (__bridge void*)output_;
    Outputs()[0]->template GetMutable<MPSImageWrapper>()->commandBuffer_ =
        (__bridge void*)commandBuffer;
    VLOG(2) << "MPSCNNFC took: " << t.MilliSeconds();
    return true;
  }

  MPSCNNConvolution* fc_{nullptr};
  MPSTemporaryImage* output_{nullptr};
};

#define INIT_FC_NEURON_OP(name, neuron)                        \
  REGISTER_CPU_OPERATOR(name, MPSCNNFullyConnectedOp<neuron>); \
  OPERATOR_SCHEMA(name).NumInputs(3).NumOutputs(1);

INIT_FC_NEURON_OP(MPSCNNFC, EmptyNeuronInit);
INIT_FC_NEURON_OP(MPSCNNFCRelu, ReluNeuronInit);
#undef INIT_FC_NEURON_OP

class MPSCNNDropoutOp final : public Operator<CPUContext> {
 public:
  MPSCNNDropoutOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  // Just pass inputs through, since we assume inference-time only.
  bool RunOnDevice() override {
    Outputs()[0]->GetMutable<MPSImageWrapper>()->x = Inputs()[0]->Get<MPSImageWrapper>().x;
    Outputs()[0]->GetMutable<MPSImageWrapper>()->commandBuffer_ =
        Inputs()[0]->Get<MPSImageWrapper>().commandBuffer_;
    return true;
  }
};

REGISTER_CPU_OPERATOR(MPSCNNDropout, MPSCNNDropoutOp);
// Never use the second output (the mask).
OPERATOR_SCHEMA(MPSCNNDropout).NumInputs(1).NumOutputs(1, 2).AllowInplace({{0, 0}});

class MPSCNNConvTransposeOp final : public ConvTransposeUnpoolBase<CPUContext> {
 public:
  MPSCNNConvTransposeOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvTransposeUnpoolBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW, "Metal only supports NCHW order.");
    CAFFE_ENFORCE_EQ(
        this->stride_h_, this->stride_w_, "Metal only supports symmetric ConvTranspose");
    CAFFE_ENFORCE_EQ(
        this->kernel_w_, this->kernel_h_, "Metal only supports symmetric ConvTranspose");
    CAFFE_ENFORCE_EQ(this->pad_t_, this->pad_l_, "Metal only supports symmetric ConvTranspose");
    CAFFE_ENFORCE_EQ(this->pad_t_, this->pad_b_, "Metal only supports symmetric ConvTranspose");
    CAFFE_ENFORCE_EQ(this->pad_t_, this->pad_r_, "Metal only supports symmetric ConvTranspose");
  }

  bool RunOnDeviceWithOrderNCHW() override {
    caffe2::Timer t;

    MPSTemporaryImage* X =
        (__bridge MPSTemporaryImage*)(Inputs()[0]->template Get<MPSImageWrapper>().x);
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)(Inputs()[0]
                                            ->template Get<MPSImageWrapper>()
                                            .commandBuffer_);

    auto& filter = Input(FILTER);
    auto& bias = Input(BIAS);
    CAFFE_ENFORCE(filter.ndim(), 4);
    const int output_channels = filter.dim32(1);
    const int input_channels = filter.dim32(0);

    CAFFE_ENFORCE(X.featureChannels == input_channels, "");
    CAFFE_ENFORCE(filter.dim32(2) == this->kernel_h_, "");
    CAFFE_ENFORCE(filter.dim32(3) == this->kernel_w_, "");
    CAFFE_ENFORCE(bias.ndim() == 1, "");
    CAFFE_ENFORCE(bias.dim32(0) == output_channels, "");

    const auto kH = this->kernel_h_;
    const auto kW = this->kernel_w_;

    if (!conv_) {
      caffe2::Timer consT;
      std::vector<float> refilter(kH * kW * output_channels * input_channels);
      refilter.assign(kH * kW * output_channels * input_channels, 0.0f);
      DCHECK_EQ(refilter.size(), filter.size());
      auto* filter_ = filter.template data<float>();
      for (auto oc = 0; oc < output_channels; ++oc) {
        for (auto ic = 0; ic < input_channels; ++ic) {
          for (auto kh = 0; kh < kH; ++kh) {
            for (auto kw = 0; kw < kW; ++kw) {
              const auto inputIdx = ic * output_channels * kH * kW + oc * kH * kW + kh * kW + kw;
              const auto outputIdx = kh * kW * output_channels * input_channels +
                                     kw * output_channels * input_channels + oc * input_channels +
                                     ic;
              DCHECK_LT(inputIdx, filter.size());
              DCHECK_LT(outputIdx, filter.size());
              refilter[outputIdx] = filter_[inputIdx];
            }
          }
        }
      }
      MPSCNNConvolutionDescriptor* desc = [MPSCNNConvolutionDescriptor
          cnnConvolutionDescriptorWithKernelWidth:1
                                     kernelHeight:1
                             inputFeatureChannels:input_channels
                            outputFeatureChannels:output_channels * kH * kW
                                     neuronFilter:nil];
      DCHECK_EQ(filter.size(), input_channels * output_channels * kH * kW);
      // We need to zero-fill the bias here.
      std::vector<float> fakeBias;
      fakeBias.assign(output_channels * kH * kW, 0);

      desc.strideInPixelsX = 1;
      desc.strideInPixelsY = 1;
      conv_ = [[MPSCNNConvolution alloc] initWithDevice:getMPSCNNContext().device
                                  convolutionDescriptor:desc
                                          kernelWeights:refilter.data()
                                              biasTerms:fakeBias.data() /* TODO: fix */
                                                  flags:MPSCNNConvolutionFlagsNone];
      [conv_ setEdgeMode:MPSImageEdgeModeZero];
      MPSOffset offset;
      offset.x = 0;
      offset.y = 0;
      offset.z = 0;
      [conv_ setOffset:offset];
      VLOG(2) << "MPSCNNConvTranspose ConvDesc took: " << consT.MilliSeconds();
    }

    CAFFE_ENFORCE_EQ(conv_.strideInPixelsY, 1);
    CAFFE_ENFORCE_EQ(conv_.strideInPixelsX, 1);
    CAFFE_ENFORCE_EQ(conv_.groups, 1);
    CAFFE_ENFORCE_EQ(conv_.inputFeatureChannels, input_channels);
    CAFFE_ENFORCE_EQ(conv_.outputFeatureChannels, output_channels * kH * kW);
    CAFFE_ENFORCE_EQ(conv_.kernelWidth, 1);
    CAFFE_ENFORCE_EQ(conv_.kernelHeight, 1);

    MPSTemporaryImage* gemmed = createImage(
        this, commandBuffer, X.numberOfImages, X.height, X.width, output_channels * kH * kW);
    {
      caffe2::Timer gt;
      [conv_ encodeToCommandBuffer:commandBuffer sourceImage:X destinationImage:gemmed];
      VLOG(2) << "MPSCNNConvTranspose GEMM took: " << gt.MilliSeconds();
    }

    int output_height =
        (X.height - 1) * this->stride_h_ + kH - this->pad_b_ - this->pad_t_ + this->adj_h_;
    int output_width =
        (X.width - 1) * this->stride_w_ + kW - this->pad_l_ - this->pad_r_ + this->adj_w_;

    VLOG(2) << "Output height: " << output_height;
    VLOG(2) << "Output width:" << output_width;
    VLOG(2) << "Output channels:" << output_channels;

    MPSTemporaryImage* col2im = createImage(
        this, commandBuffer, X.numberOfImages, output_height, output_width, output_channels);

    {
      caffe2::Timer cit;
      id<MTLComputePipelineState> state =
          getMPSCNNContext().getSpecializedPipelineState(@"col2im",
                                                         {{ushort(this->kernel_w_),
                                                           ushort(this->stride_w_),
                                                           ushort(this->pad_b_),
                                                           ushort(col2im.featureChannels),
                                                           ushort(col2im.numberOfImages),
                                                           ushort(gemmed.height),
                                                           ushort(gemmed.width)}});
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
      [encoder setComputePipelineState:state];
      [encoder setTexture:[gemmed texture] atIndex:0];
      [encoder setTexture:[col2im texture] atIndex:1];
      const auto& launchParams = spatialPointwiseKernelLaunchParams(state, col2im);
      [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
              threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
      [encoder endEncoding];
      gemmed.readCount -= 1;
      VLOG(2) << "MPSCNNConvTranspose upscaling took: " << cit.MilliSeconds();
    }

    // Apply bias.

    const auto scaleBytes = divRoundUp(bias.size(), 4) * 4 * 2;
    if (!scaleBuffer_ || scaleBuffer_.length != scaleBytes) {
      caffe2::Timer cvt;
      scaleBuffer_ =
          [getMPSCNNContext().device newBufferWithLength:scaleBytes
                                                 options:MTLResourceOptionCPUCacheModeDefault];
      shiftBuffer_ =
          [getMPSCNNContext().device newBufferWithLength:scaleBytes
                                                 options:MTLResourceOptionCPUCacheModeDefault];
      for (auto i = 0; i < bias.size(); ++i) {
        ((float16_t*)[scaleBuffer_ contents])[i] = 1.0;
        ((float16_t*)[shiftBuffer_ contents])[i] = bias.data<float>()[i];
      }
      VLOG(2) << "Buffer setup took: " << cvt.MilliSeconds();
    }

    output_ = createImage(this,
                          commandBuffer,
                          col2im.numberOfImages,
                          col2im.height,
                          col2im.width,
                          col2im.featureChannels);
    {
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
      id<MTLComputePipelineState> state = getMPSCNNContext().getSpecializedPipelineState(
          kernelFor(output_, @"affine", @"affine_nonarray"), {ushort(output_.featureChannels)});

      [encoder setComputePipelineState:state];
      [encoder setBuffer:scaleBuffer_ offset:0 atIndex:0];
      [encoder setBuffer:shiftBuffer_ offset:0 atIndex:1];
      [encoder setTexture:[col2im texture] atIndex:0];
      [encoder setTexture:[output_ texture] atIndex:1];

      const auto& launchParams = spatialPointwiseKernelLaunchParams(state, col2im);
      [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
              threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
      [encoder endEncoding];
      col2im.readCount -= 1;
    }

    Outputs()[0]->template GetMutable<MPSImageWrapper>()->x = (__bridge void*)output_;
    Outputs()[0]->template GetMutable<MPSImageWrapper>()->commandBuffer_ =
        (__bridge void*)commandBuffer;
    VLOG(2) << "MPSCNNConvTranspose took: " << t.MilliSeconds();
    return true;
  }

  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
  id<MTLBuffer> scaleBuffer_;
  id<MTLBuffer> shiftBuffer_;

  MPSCNNConvolution* conv_{nullptr};
  MPSTemporaryImage* output_{nullptr};
};

// No-op init
#define INIT_CONV_TRANSPOSE_NEURON_OP(name)           \
  REGISTER_CPU_OPERATOR(name, MPSCNNConvTransposeOp); \
  OPERATOR_SCHEMA(name).NumInputs(3).NumOutputs(1);

INIT_CONV_TRANSPOSE_NEURON_OP(MPSCNNConvTranspose);
#undef INIT_CONV_TRANSPOSE_NEURON_OP

class MPSCNNInstanceNormOp final : public Operator<CPUContext> {
 public:
  MPSCNNInstanceNormOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    const MPSTemporaryImage* X =
        (__bridge MPSTemporaryImage*)(Inputs()[0]->Get<MPSImageWrapper>().x);
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)(Inputs()[0]
                                            ->template Get<MPSImageWrapper>()
                                            .commandBuffer_);

    const auto& scale = Input(1);
    const auto& bias = Input(2);
    CAFFE_ENFORCE_EQ(scale.size(), X.featureChannels);
    CAFFE_ENFORCE_EQ(bias.size(), X.featureChannels);
    const auto scaleBytes = divRoundUp(scale.size(), 4) * 4 * 2;
    if (!scaleBuffer_ || !biasBuffer_ || scaleBuffer_.length != scaleBytes ||
        biasBuffer_.length != scaleBytes) {
      caffe2::Timer cvt;
      // Round-up to nearest multiple of 4,
      // so accesses to X[i * 4 + 3]  in kernel is valid.
      scaleBuffer_ =
          [getMPSCNNContext().device newBufferWithLength:scaleBytes
                                                 options:MTLResourceOptionCPUCacheModeDefault];
      biasBuffer_ =
          [getMPSCNNContext().device newBufferWithLength:scaleBytes
                                                 options:MTLResourceOptionCPUCacheModeDefault];
      for (auto i = 0; i < scale.size(); ++i) {
        ((float16_t*)[scaleBuffer_ contents])[i] = scale.data<float>()[i];
      }
      for (auto i = 0; i < bias.size(); ++i) {
        ((float16_t*)[biasBuffer_ contents])[i] = bias.data<float>()[i];
      }
      VLOG(2) << "Buffer setup took: " << cvt.MilliSeconds();
    }

    output_ =
        createImage(this, commandBuffer, X.numberOfImages, X.height, X.width, X.featureChannels);

    caffe2::Timer t;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state = getMPSCNNContext().getPipelineState(
        kernelFor(X, @"instance_norm", @"instance_norm_nonarray"));

    [encoder setComputePipelineState:state];
    [encoder setBuffer:scaleBuffer_ offset:0 atIndex:0];
    [encoder setBuffer:biasBuffer_ offset:0 atIndex:1];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[output_ texture] atIndex:1];

    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, divRoundUp(X.featureChannels, 4))
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    X.readCount -= 1;
    VLOG(2) << "InstanceNorm took: " << t.MilliSeconds();
    Outputs()[0]->GetMutable<MPSImageWrapper>()->x = (__bridge void*)output_;
    Outputs()[0]->template GetMutable<MPSImageWrapper>()->commandBuffer_ =
        (__bridge void*)commandBuffer;

    return true;
  }

 private:
  id<MTLBuffer> scaleBuffer_;
  id<MTLBuffer> biasBuffer_;
  MPSTemporaryImage* output_{nullptr};
};

REGISTER_CPU_OPERATOR(MPSCNNInstanceNorm, MPSCNNInstanceNormOp);
OPERATOR_SCHEMA(MPSCNNInstanceNorm).NumInputs(3).NumOutputs(1);

class MPSCNNPReluOp final : public Operator<CPUContext> {
 public:
  MPSCNNPReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    const MPSTemporaryImage* X =
        (__bridge MPSTemporaryImage*)(Inputs()[0]->Get<MPSImageWrapper>().x);
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)(Inputs()[0]
                                            ->template Get<MPSImageWrapper>()
                                            .commandBuffer_);

    const auto& scale = Input(1);
    const auto scaleBytes = divRoundUp(scale.size(), 4) * 4 * 2;
    if (!scaleBuffer_ || scaleBuffer_.length != scaleBytes) {
      caffe2::Timer cvt;
      scaleBuffer_ =
          [getMPSCNNContext().device newBufferWithLength:scaleBytes
                                                 options:MTLResourceOptionCPUCacheModeDefault];
      for (auto i = 0; i < scale.size(); ++i) {
        ((float16_t*)[scaleBuffer_ contents])[i] = scale.data<float>()[i];
      }
      VLOG(2) << "Buffer setup took: " << cvt.MilliSeconds();
    }

    output_ =
        createImage(this, commandBuffer, X.numberOfImages, X.height, X.width, X.featureChannels);
    caffe2::Timer t;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state = getMPSCNNContext().getSpecializedPipelineState(
        kernelFor(X, @"prelu_nonshared", @"prelu_nonshared_nonarray"),
        {{ushort(X.featureChannels), ushort(scale.size())}});

    [encoder setComputePipelineState:state];
    [encoder setBuffer:scaleBuffer_ offset:0 atIndex:0];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[output_ texture] atIndex:1];

    const auto& launchParams = spatialPointwiseKernelLaunchParams(state, output_);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    X.readCount -= 1;
    VLOG(2) << "PRelu took: " << t.MilliSeconds();
    Outputs()[0]->GetMutable<MPSImageWrapper>()->x = (__bridge void*)output_;
    Outputs()[0]->template GetMutable<MPSImageWrapper>()->commandBuffer_ =
        (__bridge void*)commandBuffer;

    return true;
  }

 private:
  id<MTLBuffer> scaleBuffer_;
  MPSTemporaryImage* output_{nullptr};
};

REGISTER_CPU_OPERATOR(MPSCNNPRelu, MPSCNNPReluOp);
// Allow in-place isn't *really* valid here, since nothing is in-place for Metal texture arrays,
// but requires re-export.
OPERATOR_SCHEMA(MPSCNNPRelu).NumInputs(2).NumOutputs(1).AllowInplace({{0, 0}});

class MPSCNNRoIWarpOp final : public Operator<CPUContext> {
 public:
  MPSCNNRoIWarpOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        spatial_scale_(OperatorBase::GetSingleArgument<float>("spatial_scale", 1.)),
        pooled_height_(OperatorBase::GetSingleArgument<int>("pooled_h", 1)),
        pooled_width_(OperatorBase::GetSingleArgument<int>("pooled_w", 1)),
        sampling_ratio_(OperatorBase::GetSingleArgument<int>("sampling_ratio", -1)) {
    CAFFE_ENFORCE_GT(spatial_scale_, 0);
    CAFFE_ENFORCE_GT(pooled_height_, 0);
    CAFFE_ENFORCE_GT(pooled_width_, 0);
    CAFFE_ENFORCE_GE(sampling_ratio_, 0);
    VLOG(1) << "spatial_scale: " << spatial_scale_;
    VLOG(1) << "pooled_h: " << pooled_height_;
    VLOG(1) << "pooled_w: " << pooled_width_;
    VLOG(1) << "sampling_ration: " << sampling_ratio_;
    // Skip some conditionals in the kernel.
    CAFFE_ENFORCE_GE(sampling_ratio_, 1);
  }

  bool RunOnDevice() override {
    caffe2::Timer t;

    const MPSTemporaryImage* X =
        (__bridge MPSTemporaryImage*)(Inputs()[0]->Get<MPSImageWrapper>().x);
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)(Inputs()[0]
                                            ->template Get<MPSImageWrapper>()
                                            .commandBuffer_);
    const auto& R = Input(1);
    CAFFE_ENFORCE_EQ(X.numberOfImages, 1);
    CAFFE_ENFORCE_EQ(R.ndim(), 2);
    CAFFE_ENFORCE(R.dim32(1) == 4 || R.dim32(1) == 5);
    const auto roiBytes = R.dim32(0) * 4 * sizeof(float16_t);
    if (!roiBuffer_ || roiBuffer_.length != roiBytes) {
      caffe2::Timer cvt;
      roiBuffer_ =
          [getMPSCNNContext().device newBufferWithLength:roiBytes
                                                 options:MTLResourceOptionCPUCacheModeDefault];
    }
    float16_t* roiBuffer = (float16_t*)[roiBuffer_ contents];
    // Help compiler generate vcvt?
    const auto Rdim = R.dim32(1);
    CAFFE_ENFORCE(Rdim == 4 || Rdim == 5);
    auto off = Rdim == 5 ? 1 : 0;
    for (auto i = 0; i < R.dim32(0); ++i) {
      if (Rdim == 5) {
        // only handle batch-size of one, so the batch index must be one.
        CAFFE_ENFORCE_EQ(R.data<float>()[i * Rdim], 0.0);
      }
      roiBuffer[i * 4 + 0] = R.data<float>()[i * Rdim + off + 0];
      roiBuffer[i * 4 + 1] = R.data<float>()[i * Rdim + off + 1];
      roiBuffer[i * 4 + 2] = R.data<float>()[i * Rdim + off + 2];
      roiBuffer[i * 4 + 3] = R.data<float>()[i * Rdim + off + 3];
    }

    output_ = createImage(
        this, commandBuffer, R.dim32(0), pooled_height_, pooled_width_, X.featureChannels);
    VLOG(1) << "output: " << output_.numberOfImages << ", " << output_.featureChannels << ", "
            << output_.height << ", " << output_.width;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state = getMPSCNNContext().getSpecializedPipelineState(
        kernelFor(output_, @"roi_warp", @"roi_warp_nonarray"),
        {{ushort(spatial_scale_ * 10000), ushort(sampling_ratio_), ushort(X.featureChannels)}});

    [encoder setComputePipelineState:state];
    [encoder setBuffer:roiBuffer_ offset:0 atIndex:0];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[output_ texture] atIndex:1];

    const auto& launchParams = spatialPointwiseKernelLaunchParams(state, output_);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    X.readCount -= 1;
    VLOG(2) << "RoIWarp took: " << t.MilliSeconds();
    VLOG(1) << "ROIWarp size: " << output_.numberOfImages << ", " << output_.featureChannels << ", "
            << output_.height << ", " << output_.width;
    Outputs()[0]->GetMutable<MPSImageWrapper>()->x = (__bridge void*)output_;
    Outputs()[0]->template GetMutable<MPSImageWrapper>()->commandBuffer_ =
        (__bridge void*)commandBuffer;

    return true;
  }

 private:
  float spatial_scale_;
  int pooled_height_;
  int pooled_width_;
  int sampling_ratio_;

  id<MTLBuffer> roiBuffer_;
  MPSTemporaryImage* output_{nullptr};
};

REGISTER_CPU_OPERATOR(MPSCNNRoIWarp, MPSCNNRoIWarpOp);
OPERATOR_SCHEMA(MPSCNNRoIWarp).NumInputs(2).NumOutputs(1);

class MPSCNNSpatialBNOp final : public SpatialBNOp<CPUContext> {
 public:
  MPSCNNSpatialBNOp(const OperatorDef& operator_def, Workspace* ws)
      : SpatialBNOp<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {

    const MPSTemporaryImage* X =
        (__bridge MPSTemporaryImage*)(Inputs()[0]->Get<MPSImageWrapper>().x);
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)(Inputs()[0]
                                            ->template Get<MPSImageWrapper>()
                                            .commandBuffer_);
    const auto& scale = Input(SCALE);
    const auto& bias = Input(BIAS);
    const auto& var = Input(EST_VAR);
    const auto& mean = Input(EST_MEAN);
    CAFFE_ENFORCE_EQ(scale.size(), X.featureChannels);
    CAFFE_ENFORCE_EQ(bias.size(), X.featureChannels);
    CAFFE_ENFORCE_EQ(var.size(), X.featureChannels);
    CAFFE_ENFORCE_EQ(mean.size(), X.featureChannels);

    const auto scaleBytes = divRoundUp(scale.size(), 4) * 4 * 2;
    if (!scaleBuffer_ || scaleBuffer_.length != scaleBytes) {
      caffe2::Timer cvt;
      scaleBuffer_ =
          [getMPSCNNContext().device newBufferWithLength:scaleBytes
                                                 options:MTLResourceOptionCPUCacheModeDefault];
      shiftBuffer_ =
          [getMPSCNNContext().device newBufferWithLength:scaleBytes
                                                 options:MTLResourceOptionCPUCacheModeDefault];
      for (auto i = 0; i < scale.size(); ++i) {
        // We can fuse the output computation as follows:
        //   ((x - est_mean) * (inv_var) * scale + bias
        // to
        //   (x * inv_var * scale) + (bias - est_mean * inv_var * scale)

        const auto inv_std = 1.0 / std::sqrt(var.data<float>()[i] + epsilon_);
        ((float16_t*)[scaleBuffer_ contents])[i] = scale.data<float>()[i] * inv_std;
        ((float16_t*)[shiftBuffer_ contents])[i] =
            bias.data<float>()[i] - mean.data<float>()[i] * inv_std * scale.data<float>()[i];
      }
      VLOG(2) << "Buffer setup took: " << cvt.MilliSeconds();
    }

    output_ =
        createImage(this, commandBuffer, X.numberOfImages, X.height, X.width, X.featureChannels);
    caffe2::Timer t;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state = getMPSCNNContext().getSpecializedPipelineState(
        kernelFor(output_, @"affine", @"affine_nonarray"), {ushort(X.featureChannels)});

    [encoder setComputePipelineState:state];
    [encoder setBuffer:scaleBuffer_ offset:0 atIndex:0];
    [encoder setBuffer:shiftBuffer_ offset:0 atIndex:1];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[output_ texture] atIndex:1];

    const auto& launchParams = spatialPointwiseKernelLaunchParams(state, output_);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    X.readCount -= 1;
    VLOG(2) << "SpatialBN took: " << t.MilliSeconds();
    Outputs()[0]->GetMutable<MPSImageWrapper>()->x = (__bridge void*)output_;
    Outputs()[0]->template GetMutable<MPSImageWrapper>()->commandBuffer_ =
        (__bridge void*)commandBuffer;
    return true;
  }

 private:
  id<MTLBuffer> scaleBuffer_;
  id<MTLBuffer> shiftBuffer_;

  MPSTemporaryImage* output_{nullptr};
};

REGISTER_CPU_OPERATOR(MPSCNNSpatialBN, MPSCNNSpatialBNOp);
OPERATOR_SCHEMA(MPSCNNSpatialBN).NumInputs(5).NumOutputs(1);

class MPSCNNConcatOp final : public Operator<CPUContext> {
 public:
  MPSCNNConcatOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {
    // Only handle three inputs for now.
    OPERATOR_NEEDS_FEATURE(Inputs().size() <= 4, "MPSCNNConcat only handles up to four inputs");
  }

  bool RunOnDevice() override {
    auto cb = [&](size_t i) { return Inputs()[i]->template Get<MPSImageWrapper>().commandBuffer_; };
    auto X = [&](size_t i) {
      return (__bridge MPSTemporaryImage*)Inputs()[i]->template Get<MPSImageWrapper>().x;
    };

    // C0, C1, C2, C3, N
    std::vector<ushort> channels = {{0, 0, 0, 0, ushort(X(0).numberOfImages)}};
    size_t channelCount = 0;
    for (auto i = 0; i < Inputs().size(); ++i) {
      CAFFE_ENFORCE_EQ(cb(0), cb(i));
      CAFFE_ENFORCE_EQ(X(0).height, X(i).height);
      CAFFE_ENFORCE_EQ(X(0).width, X(i).width);
      CAFFE_ENFORCE_EQ(X(i).featureChannels % 4, 0);
      channels[i] = X(i).featureChannels;
      channelCount += X(i).featureChannels;
    }
    id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)(cb(0));

    output_ = createImage(
        this, commandBuffer, X(0).numberOfImages, X(0).height, X(0).width, channelCount);
    caffe2::Timer t;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state =
        getMPSCNNContext().getSpecializedPipelineState(@"concat", channels);

    [encoder setComputePipelineState:state];
    for (auto i = 0; i < Inputs().size(); ++i) {
      [encoder setTexture:[X(i) texture] atIndex:i];
    }
    [encoder setTexture:[output_ texture] atIndex:5];
    const auto& launchParams = spatialPointwiseKernelLaunchParams(state, output_);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    for (auto i = 0; i < Inputs().size(); ++i) {
      X(i).readCount -= 1;
    }

    VLOG(2) << "Concat took: " << t.MilliSeconds();
    Outputs()[0]->GetMutable<MPSImageWrapper>()->x = (__bridge void*)output_;
    Outputs()[0]->template GetMutable<MPSImageWrapper>()->commandBuffer_ =
        (__bridge void*)commandBuffer;

    return true;
  }

 private:
  MPSTemporaryImage* output_{nullptr};
};

REGISTER_CPU_OPERATOR(MPSCNNConcat, MPSCNNConcatOp);
// Only store one output in practice (ignore the shape argument).
OPERATOR_SCHEMA(MPSCNNConcat).NumInputs(2, 4).NumOutputs(1, 2);

struct Analysis {
  struct SSA {
    using BlobVersions = std::unordered_map<std::string, size_t>;
    BlobVersions inVersions;
    BlobVersions outVersions;
  };
  std::vector<SSA> ssa;
  std::unordered_map<std::string, std::unordered_map<size_t, std::vector<size_t>>> inUsages;
};

Analysis analyzeNet(const NetDef& net) {
  Analysis::SSA::BlobVersions frontier;
  Analysis analysis;

  auto play = [&](size_t i, const OperatorDef& op) {
    Analysis::SSA::BlobVersions inVersions;
    for (const auto& s : op.input()) {
      inVersions[s] = frontier[s];
      analysis.inUsages[s][frontier[s]].push_back(i);
    }
    Analysis::SSA::BlobVersions outVersions;
    for (const auto& s : op.output()) {
      if (frontier.find(s) != frontier.end()) {
        frontier[s] += 1;
      }
      outVersions[s] = frontier[s];
    }
    analysis.ssa.push_back(Analysis::SSA{inVersions, outVersions});
  };

  for (auto i = 0; i < net.op_size(); ++i) {
    play(i, net.op(i));
  }
  return analysis;
}

NetDef insertInputOutputCopyOps(const NetDef& def) {
  // Do some validation of the outputs. For this version, we require:
  // - a single input (first element of external_input()) is consumed by the NetDef
  // - a single output (first element of external_output()) is produced by the NetDef.
  // - the input is consumed by def.op(0), and this is the only consumer.
  // - the output is produced by def.op(-1).
  CAFFE_ENFORCE_GE(def.external_input_size(), 1);
  CAFFE_ENFORCE_GE(def.external_output_size(), 1);
  auto analysis = analyzeNet(def);
  // enforce a single use of the input blob.
  CAFFE_ENFORCE_GE(def.op_size(), 1);

  const auto& inputBlob = def.external_input(0);
  // Enforce that the input blob has a single usage - in the first operator.
  CAFFE_ENFORCE(analysis.inUsages[inputBlob][0] == (std::vector<size_t>{0}));
  // Enforce that the external_output(0) blob is produced by the last operator in this sequence.
  const auto& outputBlob = def.external_output(0);
  CAFFE_ENFORCE(analysis.ssa.back().outVersions.find(outputBlob) !=
                analysis.ssa.back().outVersions.end());
  const auto& outputBlobVersion = analysis.ssa.back().outVersions[outputBlob];
  // This should hold true by definition of the SSA analysis.
  CAFFE_ENFORCE(analysis.inUsages[outputBlob].find(outputBlobVersion) ==
                analysis.inUsages[outputBlob].end());
  NetDef mdef;
  mdef.CopyFrom(def);
  mdef.clear_op();

  {
    auto& op = *(mdef.add_op());
    op.set_type("CopyToMPSCNN");
    op.add_input(def.external_input(0));
    op.add_output("__METAL_INPUT_COPY__");
  }

  for (auto i = 0; i < def.op_size(); ++i) {
    const auto& ogOp = def.op(i);
    auto op = mdef.add_op();
    op->CopyFrom(ogOp);
    if (i == 0) {
      CAFFE_ENFORCE_EQ(op->input(0), def.external_input(0));
      op->set_input(0, "__METAL_INPUT_COPY__");
    }
    if (i == def.op_size() - 1) {
      CAFFE_ENFORCE_EQ(op->output(0), def.external_output(0));
      op->set_output(0, "__METAL_OUTPUT_COPY__");
    }
  }
  {
    auto& op = *(mdef.add_op());
    op.set_type("CopyFromMPSCNN");
    op.add_input("__METAL_OUTPUT_COPY__");
    op.add_output(def.external_output(0));
  }
  return mdef;
}

bool tryFuseAdjacentOps(const OperatorDef& currentOp,
                        const OperatorDef& nextOp,
                        OperatorDef* fusedOp) {
  // Check for possible invalid opportunities.
  // Must be identical outputs, with in-place usage for nextOp.
  if (currentOp.output_size() != 1 || nextOp.output_size() != 1 || nextOp.input_size() != 1) {
    return false;
  }
  if (currentOp.output(0) != nextOp.input(0) || nextOp.input(0) != nextOp.output(0)) {
    return false;
  }

  // Can we autogenerate this at registration time instead?
  static const std::map<std::pair<std::string, std::string>, std::string> fusionOpportunities = {{
      {{"MPSCNNConv", "MPSCNNRelu"}, "MPSCNNConvRelu"},
      {{"MPSCNNConv", "MPSCNNSigmoid"}, "MPSCNNConvSigmoid"},
      {{"MPSCNNFC", "MPSCNNRelu"}, "MPSCNNFCRelu"},
  }};
  auto it = fusionOpportunities.find({currentOp.type(), nextOp.type()});
  if (it == fusionOpportunities.end()) {
    return false;
  }
  LOG(INFO) << "Found a fusion between adjacent ops: (" << currentOp.type() << ", " << nextOp.type()
            << ") -> " << it->second;
  fusedOp->CopyFrom(currentOp);
  fusedOp->set_type(it->second);
  return true;
}
}

CAFFE_KNOWN_TYPE(MPSImageWrapper);

NetDef runMPSCNNFusion(const NetDef& def) {
  CHECK_GE(def.op_size(), 1);
  NetDef mdef;
  mdef.CopyFrom(def);
  mdef.clear_op();
  auto i = 0;

  while (i < def.op_size()) {
    if (i == def.op_size() - 1) {
      VLOG(2) << "Last operator, skipping";
      auto* op = mdef.add_op();
      op->CopyFrom(def.op(i));
      i += 1;
      continue;
    }

    const auto& currentOp = def.op(i);
    const auto& nextOp = def.op(i + 1);
    OperatorDef fusedOp;
    if (tryFuseAdjacentOps(currentOp, nextOp, &fusedOp)) {
      VLOG(2) << "Found an adjacent fusion at: " << i;
      // We can fuse.
      auto* op = mdef.add_op();
      op->CopyFrom(fusedOp);
      i += 2;
      continue;
    }
    VLOG(2) << "No fusion available";
    // Just emit the current type.
    auto* op = mdef.add_op();
    op->CopyFrom(currentOp);
    i += 1;
  }
  return mdef;
}

NetDef rewriteForMetal(const NetDef& def) {
  NetDef mdef;
  mdef.CopyFrom(def);

  const auto& opKeyList = CPUOperatorRegistry()->Keys();
  const auto& opKeySet = std::set<std::string>(opKeyList.begin(), opKeyList.end());
  for (auto i = 0; i < mdef.op_size(); ++i) {
    auto* op = mdef.mutable_op(i);
    const auto mpscnnOp = std::string("MPSCNN") + op->type();
    CAFFE_ENFORCE(opKeySet.find(mpscnnOp) != opKeySet.end());
    op->set_type(mpscnnOp);
  }

  mdef = runMPSCNNFusion(mdef);
  static std::set<std::string> mpscnnInputOps = {
      "CopyToMPSCNN", "MPSCNNPackedInt8BGRANHWCToNCHWCStylizerPreprocess"};
  static std::set<std::string> mpscnnOutputOps = {
      "CopyFromMPSCNN", "MPSCNNBRGNCHWCToPackedInt8BGRAStylizerDeprocess"};

  if (mpscnnInputOps.find(mdef.op(0).type()) == mpscnnInputOps.end() &&
      mpscnnOutputOps.find(mdef.op(mdef.op_size() - 1).type()) == mpscnnOutputOps.end()) {
    mdef = insertInputOutputCopyOps(mdef);
  }
  CAFFE_ENFORCE_GE(mdef.op_size(), 2);
  CAFFE_ENFORCE(mpscnnInputOps.find(mdef.op(0).type()) != mpscnnInputOps.end());
  CAFFE_ENFORCE(mpscnnOutputOps.find(mdef.op(mdef.op_size() - 1).type()) != mpscnnOutputOps.end());
  return mdef;
}

void dumpDef(const NetDef& d) {
  for (const auto& op : d.op()) {
    LOG(INFO) << op.input(0) << " -> " << op.type() << " -> " << op.output(0);
  }
}

NetDef annotateDefWithReadCounts(const NetDef& net) {
  // Now we have usage versions, we want to compute, for each blob version, the number of usages of
  // each blob version.
  // ReadCount
  auto analysis = analyzeNet(net);
  using ReadCount = std::unordered_map<std::string, size_t>;
  std::vector<ReadCount> readCounts;

  auto computeReadCount = [&](size_t i, const OperatorDef& op) {
    ReadCount rcs;
    for (const auto bv : analysis.ssa[i].outVersions) {
      const auto versionUsages = analysis.inUsages[bv.first][bv.second];
      rcs[bv.first] = versionUsages.size();
    }
    readCounts.push_back(rcs);
  };
  for (auto i = 0; i < net.op_size(); ++i) {
    computeReadCount(i, net.op(i));
  }

  NetDef annotatedNet;
  annotatedNet.CopyFrom(net);
  for (auto i = 0; i < annotatedNet.op_size(); ++i) {
    auto* op = annotatedNet.mutable_op(i);
    // TODO - relax this? CAFFE_ENFORCE_EQ(op->output_size(), 1);
    const auto& blob = op->output(0);
    const size_t readCount = readCounts[i][blob];
    if (readCount > 1) {
      auto* arg = op->add_arg();
      arg->set_name(kMPSCNNReadCountArg);
      arg->set_i(readCount);
      LOG(INFO) << "Op: " << i << ", ty: " << op->type() << ", blob: " << blob
                << ", read count: " << readCount;
    }
  }
  return annotatedNet;
}

bool tryConvertToMPSCNN(const NetDef& initNet, const NetDef& predictNet, NetDef* metalPredictNet) {
// iOS 10.0 and above.
#define SYSTEM_VERSION_GREATER_THAN_OR_EQUAL_TO(v)                                 \
  ([[[UIDevice currentDevice] systemVersion] compare:v options:NSNumericSearch] != \
   NSOrderedAscending)

  if (!SYSTEM_VERSION_GREATER_THAN_OR_EQUAL_TO(@"10.0")) {
    LOG(ERROR) << "The iOS version is < 10.0, so MPSCNN is not available";
    return false;
  }
#undef SYSTEM_VERSION_GREATER_THAN_OR_EQUAL_TO

  // The iOS GPU Family 3 v2 feature set. Introduced with the Apple A9 GPU and iOS 10.0.
  // Don't instantiate the MPSCNNContext, as that compiles the kernel source.
  if (![MTLCreateSystemDefaultDevice() supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily3_v2]) {
    LOG(ERROR) << "The iOS GPU is less than an A9, so MPSCNN is not available";
    return false;
  }

  try {
    // Instantiating the net and catching failures allows us to
    Workspace ws;
    ws.RunNetOnce(initNet);
    // Throws if unsupported operators are found.
    *metalPredictNet = rewriteForMetal(predictNet);
    *metalPredictNet = annotateDefWithReadCounts(*metalPredictNet);
    // Throws if unsupported parameters are found.
    ws.CreateNet(*metalPredictNet);
    LOG(INFO) << "MPSCNN is successfully enabled";
    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Caught exception trying to convert NetDef to MPSCNN: " << e.what();
    return false;
  }
}

void mpscnnRecordExecutionFinish() { [getMPSCNNContext().commandQueue insertDebugCaptureBoundary]; }

} // namespace caffe2

#endif
