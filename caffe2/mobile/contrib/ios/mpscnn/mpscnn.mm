#include "caffe2/core/common.h"
#include "caffe2/core/context.h"

#if defined(CAFFE2_USE_MPSCNN) && CAFFE2_MOBILE

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/conv_transpose_unpool_op_base.h"
#include "caffe2/operators/generate_proposals_op.h"
#include "caffe2/operators/generate_proposals_op_util_boxes.h"
#include "caffe2/operators/spatial_batch_norm_op.h"

#include "mpscnn.h"
#include "mpscnn_context.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <UIKit/UIDevice.h>

#define SYSTEM_VERSION_GREATER_THAN_OR_EQUAL_TO(v) \
  ([[[UIDevice currentDevice] systemVersion]       \
       compare:v                                   \
       options:NSNumericSearch] != NSOrderedAscending)

// Only compiles against Base SDK iOS 11.0 or greater
@interface ConvDataSource : NSObject<MPSCNNConvolutionDataSource>
@property float* weights_;
@property float* bias_;
@property MPSCNNConvolutionDescriptor* desc_;
@end

@implementation ConvDataSource
- (id)initWithWeight:(float*)weights
                bias:(float*)bias
                desc:(MPSCNNConvolutionDescriptor*)desc {
  self = [super init];
  self.weights_ = weights;
  self.bias_ = bias;
  self.desc_ = desc;
  return self;
}
- (float*)biasTerms {
  return self.bias_;
}

- (MPSDataType)dataType {
  return MPSDataTypeFloat32;
}
- (MPSCNNConvolutionDescriptor*)descriptor {
  return self.desc_;
}
- (NSString*)label {
  return nullptr;
}
- (BOOL)load {
  return true;
}
- (float*)lookupTableForUInt8Kernel {
  return nullptr;
}
- (void)purge {
  return;
}
- (vector_float2*)rangesForUInt8Kernel {
  return nullptr;
}
- (void*)weights {
  return self.weights_;
}
@end

namespace caffe2 {

namespace {
auto divRoundUp(uint x, uint y) -> uint {
  return (x + y - 1) / y;
}

MPSTemporaryImage* createTemporaryImage(
    const OperatorBase* op,
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
                              imageDescriptorWithChannelFormat:
                                  MPSImageFeatureChannelFormatFloat16
                                                         width:width
                                                        height:height
                                               featureChannels:channels
                                                numberOfImages:n
                                                         usage:
                                                             MTLTextureUsageShaderRead |
                                                         MTLTextureUsageShaderWrite]];
  // We'll try to look at the per-output_idx read-count argument, otherwise,
  // we'll use the operator-global default.
  const auto& readCounts = op->GetRepeatedArgument<int>(kMPSCNNReadCountArg);
  const auto readCount = readCounts.size()
      ? readCounts.at(output_idx)
      : op->GetSingleArgument<int>(kMPSCNNReadCountArg, 1);
  CAFFE_ENFORCE_GE(readCount, 1);
  image.readCount = readCount;
  return image;
}

MPSImage* createStaticImage(int n, int height, int width, int channels) {
  return [[MPSImage alloc]
       initWithDevice:getMPSCNNContext().device
      imageDescriptor:
          [MPSImageDescriptor
              imageDescriptorWithChannelFormat:
                  MPSImageFeatureChannelFormatFloat16
                                         width:width
                                        height:height
                               featureChannels:channels
                                numberOfImages:n
                                         usage:MTLTextureUsageShaderRead |
                                         MTLTextureUsageShaderWrite]];
}

class MPSImageWrapper {
 public:
  MPSImageWrapper() {}
  MPSImageWrapper(
      const OperatorBase* op,
      MPSImageWrapper* parent,
      int n,
      int height,
      int width,
      int channels,
      size_t output_idx = 0) {
    /* If the parent wrapper contains a temporary image, we need to pass on the
     * command buffer because the temporary images are attached to the command
     * buffer, we will need to use the same command buffer in order to use the
     * temporary image. We don't want to synchronize the parent wrapper because
     * it is still in use. If the parent wrapper contains a static image, we
     * should create a new command buffer because we use static image so it can
     * survive synchronization(commit of the command buffer), which means if we
     * pass on the command buffer the command buffer will be commited in
     * multiple places in the graph. Also since we don't pass on parent's
     * command buffer,we need to synchronize(commit) it since it won't be used
     * in the future.
     */
    bool passOnCb = parent != nullptr && parent->isTemporaryImage_;
    commandBuffer_ = passOnCb ? parent->commandBuffer_
                              : [getMPSCNNContext().commandQueue commandBuffer];

    bool commitInputCb = parent != nullptr && !parent->isTemporaryImage_;
    if (commitInputCb) {
      parent->synchronize();
    }

    const auto& isTemporaryImages =
        op->GetRepeatedArgument<int>(kMPSCNNOutputIsTempImageArg);
    isTemporaryImage_ = isTemporaryImages.size()
        ? isTemporaryImages.at(output_idx)
        : op->GetSingleArgument<int>(kMPSCNNOutputIsTempImageArg, 1);
    if (isTemporaryImage_) {
      image_ = createTemporaryImage(
          op, commandBuffer_, n, height, width, channels, output_idx);
    } else {
      image_ = createStaticImage(n, height, width, channels);
    }
  }

  void markRead() {
    if (isTemporaryImage_) {
      MPSTemporaryImage* tempImg = (MPSTemporaryImage*)image_;
      tempImg.readCount -= 1;
    }
  }

  MPSImage* getImage() const {
    return image_;
  }

  id<MTLCommandBuffer> getCommandBuffer() const {
    return commandBuffer_;
  }

  void synchronize() {
    // commit the command buffer if it is notEnqueued
    if (commandBuffer_ != nullptr && commandBuffer_.status == 0) {
      [commandBuffer_ commit];
    }
  }

  void cleanup() {
    markRead();
    synchronize();
  }

  void copyToOutputBlob(Blob* output) {
    output->GetMutable<MPSImageWrapper>()->image_ = image_;
    output->GetMutable<MPSImageWrapper>()->commandBuffer_ = commandBuffer_;
    output->GetMutable<MPSImageWrapper>()->isTemporaryImage_ =
        isTemporaryImage_;
  }

 private:
  MPSImage* image_{nullptr};
  id<MTLCommandBuffer> commandBuffer_{nullptr};
  bool isTemporaryImage_ = true;
};

NSString*
kernelFor(const MPSImage* X, NSString* arrayKernel, NSString* nonArrayKernel) {
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

LaunchParams spatialPointwiseKernelLaunchParams(
    id<MTLComputePipelineState> pipeline,
    const MPSImage* im) {
  const auto maxThreadsPerThreadgroup =
      [pipeline maxTotalThreadsPerThreadgroup];
  const auto threadExecutionWidth = [pipeline threadExecutionWidth];
  const auto threadsPerThreadgroup = MTLSizeMake(
      8 /* threadExecutionWidth */,
      4 /* maxThreadsPerThreadgroup / threadExecutionWidth */,
      1);
  const auto threadgroupsPerGrid = MTLSizeMake(
      divRoundUp(im.width, threadsPerThreadgroup.width),
      divRoundUp(im.height, threadsPerThreadgroup.height),
      im.numberOfImages * divRoundUp(im.featureChannels, 4));
  return {threadsPerThreadgroup, threadgroupsPerGrid};
};

void computeOutputHW(
    ConvPoolOpBase<CPUContext>* op,
    int H,
    int W,
    int* OH,
    int* OW) {
  Tensor<CPUContext> input, output;
  input.Resize(1, 1, H, W);
  op->SetOutputSize<CPUContext>(input, &output, 1);
  CAFFE_ENFORCE_EQ(output.ndim(), 4);
  *OH = output.dim(2);
  *OW = output.dim(3);
}

constexpr int computeMPSAlignOffset(int kernel, int pad) {
  // To set the offset, we can just match the top-left pixel (in the input
  // image, with negative values for padding) that we look at. For 3x3s1p1, we
  // look at the (-1, -1) pixel in the original impl. For 3x3s1p0, we look at
  // (0, 0) pixel. For 3x3s1p2, look at (-2, -2) MPSCNN always looks at
  // (-floor(kernel_size - 1 / 2), -floor(kernel_size - 1 / 2)) Thus, we just
  // need to match this up.

  // For 3x3s1p1, offset should be (0, 0)
  // For 3x3s1p0, offset should be (1, 1)
  // For 3x3s1p2, offset should be (-1, -1)
  const int mps_offset = kernel / 2;
  const int c2_offset = pad;
  return mps_offset - c2_offset;
};

// Compute the 1-d index of a n-dimensional contiguous row-major tensor for
//     a given n-dimensional index 'index'
size_t ComputeStartIndex(
    const TensorCPU& tensor,
    const std::vector<int>& index) {
  DCHECK_EQ(index.size(), tensor.ndim());

  size_t ret = 0;
  for (int i = 0; i < index.size(); i++) {
    ret += index[i] * tensor.size_from_dim(i + 1);
  }

  return ret;
}

// Get a sub tensor view from 'tensor' using data pointer from 'tensor'
template <class T>
utils::ConstTensorView<T> GetSubTensorView(
    const TensorCPU& tensor,
    int dim0_start_index) {
  DCHECK_EQ(tensor.meta().itemsize(), sizeof(T));

  if (tensor.size() == 0) {
    return utils::ConstTensorView<T>(nullptr, {});
  }

  std::vector<int> start_dims(tensor.ndim(), 0);
  start_dims.at(0) = dim0_start_index;
  auto st_idx = ComputeStartIndex(tensor, start_dims);
  auto ptr = tensor.data<T>() + st_idx;

  auto& input_dims = tensor.dims();
  std::vector<int> ret_dims(input_dims.begin() + 1, input_dims.end());

  utils::ConstTensorView<T> ret(ptr, ret_dims);
  return ret;
}

class CopyToMPSCNNOp final : public Operator<CPUContext> {
 public:
  CopyToMPSCNNOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    inputBuffers_.resize(Inputs().size());
    std::vector<MPSImageWrapper> wrappers(Inputs().size());
    for (auto i = 0; i < Inputs().size(); ++i) {
      const auto& X = Input(i);
      CAFFE_ENFORCE(X.ndim() > 0 && X.ndim() <= 4);
      std::vector<TIndex> XDims = {1, 1, 1, 1};
      XDims.assign(X.dims().begin(), X.dims().end());

      caffe2::Timer t;
      const auto n = XDims[0];
      const auto width = XDims[3];
      const auto height = XDims[2];
      const auto channels = XDims[1];
      caffe2::Timer copyT;
      if (!inputBuffers_[i] || inputBuffers_[i].length != X.nbytes()) {
        inputBuffers_[i] = [getMPSCNNContext().device
            newBufferWithLength:X.nbytes()
                        options:MTLResourceOptionCPUCacheModeWriteCombined];
      }
      memcpy([inputBuffers_[i] contents], X.raw_data(), X.nbytes());
      VLOG(2) << "CopyToMPSCNNOp input copy took: " << copyT.MilliSeconds();
      if (i == 0) {
        wrappers[i] =
            MPSImageWrapper(this, nullptr, n, height, width, channels, i);
      } else {
        wrappers[i] =
            MPSImageWrapper(this, &wrappers[0], n, height, width, channels, i);
      }
      auto commandBuffer = wrappers[i].getCommandBuffer();
      MPSImage* output = wrappers[i].getImage();
      id<MTLComputeCommandEncoder> encoder =
          [commandBuffer computeCommandEncoder];
      id<MTLComputePipelineState> state =
          getMPSCNNContext().getSpecializedPipelineState(
              kernelFor(
                  output,
                  @"copy_nchw_to_metal",
                  @"copy_nchw_to_metal_nonarray"),
              {{ushort(channels), ushort(height), ushort(width)}});
      [encoder setComputePipelineState:state];
      [encoder setBuffer:inputBuffers_[i] offset:0 atIndex:0];
      [encoder setTexture:[output texture] atIndex:0];
      const auto& launchParams =
          spatialPointwiseKernelLaunchParams(state, output);
      [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
              threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
      [encoder endEncoding];
      VLOG(2) << "CopyToMPSCNNOp took: " << t.MilliSeconds();
      wrappers[i].copyToOutputBlob(Outputs()[i]);
    }
    return true;
  }

 private:
  std::vector<id<MTLBuffer>> inputBuffers_;
};

REGISTER_CPU_OPERATOR(CopyToMPSCNN, CopyToMPSCNNOp);
OPERATOR_SCHEMA(CopyToMPSCNN)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, INT_MAX)
    .SameNumberOfOutput();

auto mpsImageSize = [](MPSImage* X) {
  return X.featureChannels * X.height * X.width * X.numberOfImages;
};

class CopyFromMPSCNNOp final : public Operator<CPUContext> {
 public:
  CopyFromMPSCNNOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    caffe2::Timer t;
    auto Wrapper = [&](size_t i) {
      return Inputs()[i]->template Get<MPSImageWrapper>();
    };
    auto cb = [&](size_t i) { return Wrapper(i).getCommandBuffer(); };
    auto X = [&](size_t i) { return Wrapper(i).getImage(); };

    auto cb0 = cb(0);
    outputBuffers_.resize(Inputs().size());
    for (auto i = 0; i < Inputs().size(); ++i) {
      CAFFE_ENFORCE_EQ(cb0, cb(i));
      MPSImage* Xi = X(i);
      if (!outputBuffers_[i] ||
          outputBuffers_[i].length != mpsImageSize(Xi) * sizeof(float)) {
        outputBuffers_[i] = [getMPSCNNContext().device
            newBufferWithLength:mpsImageSize(Xi) * sizeof(float)
                        options:MTLResourceOptionCPUCacheModeDefault];
      }
      id<MTLComputeCommandEncoder> encoder = [cb0 computeCommandEncoder];
      id<MTLComputePipelineState> state =
          getMPSCNNContext().getSpecializedPipelineState(
              kernelFor(
                  Xi, @"copy_metal_to_nchw", @"copy_metal_to_nchw_nonarray"),
              {{ushort(Xi.featureChannels),
                ushort(Xi.height),
                ushort(Xi.width)}});

      [encoder setComputePipelineState:state];
      [encoder setBuffer:outputBuffers_[i] offset:0 atIndex:0];
      [encoder setTexture:[Xi texture] atIndex:0];

      const auto& launchParams = spatialPointwiseKernelLaunchParams(state, Xi);
      [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
              threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
      [encoder endEncoding];
      Wrapper(i).markRead();
    }
    [cb0 commit];
    [cb0 waitUntilCompleted];

    for (auto i = 0; i < Inputs().size(); ++i) {
      caffe2::Timer copyOutT;
      MPSImage* Xi = X(i);
      Output(i)->Resize(
          Xi.numberOfImages, Xi.featureChannels, Xi.height, Xi.width);
      Output(i)->mutable_data<float>();
      CAFFE_ENFORCE_EQ(outputBuffers_[i].length, Output(i)->nbytes());
      memcpy(
          Output(i)->mutable_data<float>(),
          [outputBuffers_[i] contents],
          outputBuffers_[i].length);
      VLOG(2) << "CopyFromMPSCNNOp memcpy took: " << copyOutT.MilliSeconds();
    }
    VLOG(2) << "CopyFromMPSCNNOp took: " << t.MilliSeconds();
    return true;
  }

 private:
  std::vector<id<MTLBuffer>> outputBuffers_;
};

REGISTER_CPU_OPERATOR(CopyFromMPSCNN, CopyFromMPSCNNOp);
OPERATOR_SCHEMA(CopyFromMPSCNN)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, INT_MAX)
    .SameNumberOfOutput();

class MPSCNNPackedInt8BGRANHWCToNCHWCStylizerPreprocessOp final
    : public Operator<CPUContext> {
 public:
  MPSCNNPackedInt8BGRANHWCToNCHWCStylizerPreprocessOp(
      const OperatorDef& operator_def,
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
    ushort noiseSize = OperatorBase::GetSingleArgument<int>(
        "noise_size", 491 /* prime to avoid artifacts */);
    // Treaded as half4 in the kernel, so need half4 here.
    noiseSize = divRoundUp(noiseSize, 4) * 4;
    if (!noiseBlob->IsType<TensorCPU>() ||
        noiseBlob->Get<TensorCPU>().size() != noiseSize) {
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
      for (auto i = 0; i < noise.size(); ++i) {
        noiseBufferPtr[i] = noise.data<float>()[i];
      }

      VLOG(2) << "Preprocess construct took: " << pt.MilliSeconds();
    }

    {
      caffe2::Timer ct;
      memcpy([inputBuffer_ contents], X.raw_data(), X.nbytes());
      VLOG(2) << "Preprocess memcpy took: " << ct.MilliSeconds();
    }
    auto outputWrapper = MPSImageWrapper(this, nullptr, 1, H, W, 3);
    auto commandBuffer = outputWrapper.getCommandBuffer();
    MPSImage* output = outputWrapper.getImage();

    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state =
        getMPSCNNContext().getSpecializedPipelineState(
            @"preprocess_stylizer", {noiseSize});

    [encoder setComputePipelineState:state];
    [encoder setBuffer:inputBuffer_ offset:0 atIndex:0];
    [encoder setBuffer:meanBuffer_ offset:0 atIndex:1];
    [encoder setBuffer:noiseBuffer_ offset:0 atIndex:2];

    [encoder setTexture:[output texture] atIndex:0];
    const auto& launchParams =
        spatialPointwiseKernelLaunchParams(state, output);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    outputWrapper.copyToOutputBlob(Outputs()[0]);

    VLOG(2) << "Preprocess took: " << t.MilliSeconds();
    return true;
  }

 private:
  Workspace* ws_{nullptr};
  id<MTLBuffer> inputBuffer_{nullptr};
  id<MTLBuffer> noiseBuffer_{nullptr};
  id<MTLBuffer> meanBuffer_{nullptr};
};

REGISTER_CPU_OPERATOR(
    MPSCNNPackedInt8BGRANHWCToNCHWCStylizerPreprocess,
    MPSCNNPackedInt8BGRANHWCToNCHWCStylizerPreprocessOp);
OPERATOR_SCHEMA(MPSCNNPackedInt8BGRANHWCToNCHWCStylizerPreprocess)
    .NumInputs(2)
    .NumOutputs(1);

class MPSCNNBRGNCHWCToPackedInt8BGRAStylizerDeprocessOp final
    : public Operator<CPUContext> {
 public:
  MPSCNNBRGNCHWCToPackedInt8BGRAStylizerDeprocessOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    auto inputWrapper = Inputs()[0]->Get<MPSImageWrapper>();
    MPSImage* X = inputWrapper.getImage();
    id<MTLCommandBuffer> commandBuffer = inputWrapper.getCommandBuffer();

    const auto& mean = Input(1);
    caffe2::Timer t;
    const auto W = X.width;
    const auto H = X.height;
    CAFFE_ENFORCE_EQ(X.featureChannels, 3);
    CAFFE_ENFORCE_EQ(X.numberOfImages, 1);

    if (!outputBuffer_ || outputBuffer_.length != X.height * X.width * 4) {
      caffe2::Timer pt;

      outputBuffer_ = [getMPSCNNContext().device
          newBufferWithLength:X.height * X.width * 4
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
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state =
        getMPSCNNContext().getPipelineState(@"deprocess_stylizer");

    CAFFE_ENFORCE_EQ(outputBuffer_.length, X.height * X.width * 4);
    [encoder setComputePipelineState:state];
    [encoder setBuffer:outputBuffer_ offset:0 atIndex:0];
    [encoder setBuffer:meanBuffer_ offset:0 atIndex:1];
    [encoder setTexture:[X texture] atIndex:0];
    const auto& launchParams = spatialPointwiseKernelLaunchParams(state, X);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    inputWrapper.markRead();

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    Output(0)->Resize(1, X.height, X.width, 4);
    {
      caffe2::Timer ct;
      memcpy(
          Output(0)->mutable_data<uint8_t>(),
          [outputBuffer_ contents],
          [outputBuffer_ length]);
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

REGISTER_CPU_OPERATOR(
    MPSCNNBRGNCHWCToPackedInt8BGRAStylizerDeprocess,
    MPSCNNBRGNCHWCToPackedInt8BGRAStylizerDeprocessOp);
OPERATOR_SCHEMA(MPSCNNBRGNCHWCToPackedInt8BGRAStylizerDeprocess)
    .NumInputs(2)
    .NumOutputs(1);

template <typename Neuron>
class MPSCNNNeuronOp final : public Operator<CPUContext> {
 public:
  MPSCNNNeuronOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    caffe2::Timer t;
    auto inputWrapper = Inputs()[0]->template Get<MPSImageWrapper>();
    MPSImage* X = inputWrapper.getImage();

    auto outputWrapper = MPSImageWrapper(
        this,
        &inputWrapper,
        X.numberOfImages,
        X.height,
        X.width,
        X.featureChannels);
    auto commandBuffer = outputWrapper.getCommandBuffer();
    MPSImage* output = outputWrapper.getImage();
    CAFFE_ENFORCE_EQ(output.width, X.width);
    CAFFE_ENFORCE_EQ(output.height, X.height);
    CAFFE_ENFORCE_EQ(output.featureChannels, X.featureChannels);

    if (!neuron_) {
      neuron_ = Neuron::t();
    }
    [neuron_ encodeToCommandBuffer:commandBuffer
                       sourceImage:X
                  destinationImage:output];
    outputWrapper.copyToOutputBlob(Outputs()[0]);

    VLOG(2) << "ElementwiseAdd took: " << t.MilliSeconds();
    return true;
  }
  MPSCNNNeuron* neuron_{nullptr};
};

#define INIT_NEURON_OP(n)                                          \
  REGISTER_CPU_OPERATOR(MPSCNN##n, MPSCNNNeuronOp<n##NeuronInit>); \
  OPERATOR_SCHEMA(MPSCNN##n).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});

struct SigmoidNeuronInit {
  static MPSCNNNeuron* t() {
    return
        [[MPSCNNNeuronSigmoid alloc] initWithDevice:getMPSCNNContext().device];
  }
};
INIT_NEURON_OP(Sigmoid);

struct ReluNeuronInit {
  static MPSCNNNeuron* t() {
    return
        [[MPSCNNNeuronReLU alloc] initWithDevice:getMPSCNNContext().device a:0];
  }
};
INIT_NEURON_OP(Relu);

struct TanhNeuronInit {
  static MPSCNNNeuron* t() {
    return [[MPSCNNNeuronTanH alloc] initWithDevice:getMPSCNNContext().device
                                                  a:1
                                                  b:1];
  }
};
INIT_NEURON_OP(Tanh);

#undef INIT_NEURON_OP

template <typename Neuron>
class MPSCNNConvOp final : public ConvPoolOpBase<CPUContext> {
 public:
  MPSCNNConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NCHW, "Metal only supports NCHW order.");
    OPERATOR_NEEDS_FEATURE(
        kernel_h() == kernel_w(),
        "Metal only supports equal kernel dimension.");
  }

  bool RunOnDeviceWithOrderNCHW() override {
    caffe2::Timer t;
    auto inputWrapper = Inputs()[0]->template Get<MPSImageWrapper>();
    MPSImage* X = inputWrapper.getImage();

    auto& filter = Input(FILTER);
    auto& bias = Input(BIAS);
    CAFFE_ENFORCE_EQ(filter.ndim(), 4);
    // For NCHW, X.dim32(1), inputChannels
    const int C = X.featureChannels;
    const int M = filter.dim32(0);
    const int Cf = filter.dim32(1);

    CAFFE_ENFORCE(filter.dim32(2) == kernel_h(), "");
    CAFFE_ENFORCE(filter.dim32(3) == kernel_w(), "");
    CAFFE_ENFORCE(bias.ndim() == 1, "");
    CAFFE_ENFORCE(bias.dim32(0) == M, "");

    const auto kH = kernel_h();
    const auto kW = kernel_w();

    // ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));
    // Reformat weights from [M][C][kH][kW] to [M][kH][kW][C].
    if (!conv_) {
      caffe2::Timer consT;
      std::vector<float> refilter(M * kH * kW * Cf);
      auto* filter_ = filter.template data<float>();
      for (auto m = 0; m < M; ++m) {
        for (auto c = 0; c < Cf; ++c) {
          for (auto kh = 0; kh < kH; ++kh) {
            for (auto kw = 0; kw < kW; ++kw) {
              // refilter[m][kh][kw][c]
              refilter[m * kH * kW * Cf + kh * kW * Cf + kw * Cf + c] =
                  // filter[m][c][kh][kw]
                  filter_[m * Cf * kH * kW + c * kH * kW + kh * kW + kw];
            }
          }
        }
      }
      // DepthwiseConv path
      bool runtimeAtLeastIOS11 =
          SYSTEM_VERSION_GREATER_THAN_OR_EQUAL_TO(@"11.0");
      // Only inputFeatureChannels == outputFeatureChannels is supported right
      // now
      if (runtimeAtLeastIOS11 && this->group_ > 1 && Cf == 1 &&
          M == this->group_) {
        MPSCNNDepthWiseConvolutionDescriptor* desc =
            [MPSCNNDepthWiseConvolutionDescriptor
                cnnConvolutionDescriptorWithKernelWidth:kW
                                           kernelHeight:kH
                                   inputFeatureChannels:C
                                  outputFeatureChannels:M
                                           neuronFilter:Neuron::t()];
        desc.strideInPixelsX = stride_w();
        desc.strideInPixelsY = stride_h();
        desc.groups = 1;
        auto data_source = [[ConvDataSource alloc]
            initWithWeight:refilter.data()
                      bias:const_cast<float*>(bias.template data<float>())
                      desc:desc];
        conv_ =
            [[MPSCNNConvolution alloc] initWithDevice:getMPSCNNContext().device
                                              weights:data_source];
      } else {
        if (this->group_ > 1) {
          CAFFE_ENFORCE_EQ(
              Cf % 4,
              0,
              "MPSCNNConvolution requires number of input \
                           channels in each group to be multiple of 4 for \
                           group > 1.");
        }
        MPSCNNConvolutionDescriptor* desc = [MPSCNNConvolutionDescriptor
            cnnConvolutionDescriptorWithKernelWidth:kW
                                       kernelHeight:kH
                               inputFeatureChannels:C
                              outputFeatureChannels:M
                                       neuronFilter:Neuron::t()];
        desc.strideInPixelsX = stride_w();
        desc.strideInPixelsY = stride_h();
        desc.groups = this->group_;
        auto data_source = [[ConvDataSource alloc]
            initWithWeight:refilter.data()
                      bias:const_cast<float*>(bias.template data<float>())
                      desc:desc];
        conv_ =
            [[MPSCNNConvolution alloc] initWithDevice:getMPSCNNContext().device
                                              weights:data_source];
      }

      [conv_ setEdgeMode:MPSImageEdgeModeZero];

      MPSOffset offset;
      offset.x = computeMPSAlignOffset(kW, pad_l());
      offset.y = computeMPSAlignOffset(kH, pad_t());
      offset.z = 0;
      [conv_ setOffset:offset];
      VLOG(2) << "MPSCNNConv ConvDesc took: " << consT.MilliSeconds();
    }

    CAFFE_ENFORCE_EQ(conv_.strideInPixelsY, stride_h());
    CAFFE_ENFORCE_EQ(conv_.strideInPixelsX, stride_w());
    CAFFE_ENFORCE_EQ(conv_.inputFeatureChannels, Cf * this->group_);
    CAFFE_ENFORCE_EQ(M % conv_.groups, 0);
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
    auto outputWrapper = MPSImageWrapper(
        this,
        &inputWrapper,
        X.numberOfImages,
        output_height,
        output_width,
        output_channels);
    auto commandBuffer = outputWrapper.getCommandBuffer();
    MPSImage* output = outputWrapper.getImage();
    CAFFE_ENFORCE_EQ(output.height, output_height);
    CAFFE_ENFORCE_EQ(output.width, output_width);
    [conv_ encodeToCommandBuffer:commandBuffer
                     sourceImage:X
                destinationImage:output];
    outputWrapper.copyToOutputBlob(Outputs()[0]);

    VLOG(2) << "MPSCNNConv took: " << t.MilliSeconds();
    return true;
  }

  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);

  MPSCNNConvolution* conv_{nullptr};
};

// No-op init
struct EmptyNeuronInit {
  static MPSCNNNeuron* t() {
    return nil;
  }
};

// We can allow the input weights/bias and output to alias each other,
// for example when doing a Conv + out-of-place ReLU, then fusing.
#define INIT_CONV_NEURON_OP(name, neuron)                        \
  REGISTER_CPU_OPERATOR(name, MPSCNNConvOp<neuron>);             \
  OPERATOR_SCHEMA(name).NumInputs(3).NumOutputs(1).AllowInplace( \
      {{1, 0}, {2, 0}});

INIT_CONV_NEURON_OP(MPSCNNConv, EmptyNeuronInit);
INIT_CONV_NEURON_OP(MPSCNNConvRelu, ReluNeuronInit);
INIT_CONV_NEURON_OP(MPSCNNConvSigmoid, SigmoidNeuronInit);

#undef INIT_CONV_NEURON_OP

class MPSCNNPadImageOp final : public ConvPoolOpBase<CPUContext> {
 public:
  MPSCNNPadImageOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NCHW, "Metal only supports NCHW order.");

    OPERATOR_NEEDS_FEATURE(
        OperatorBase::GetSingleArgument<string>("mode", "") == "reflect",
        "Metal only supports reflection");
    kernel_[0] = kernel_[1] = 1;
  }

  bool RunOnDeviceWithOrderNCHW() override {
    caffe2::Timer t;
    auto inputWrapper = Inputs()[0]->Get<MPSImageWrapper>();
    MPSImage* X = inputWrapper.getImage();

    const auto pH = pad_t();
    const auto pW = pad_l();
    const auto output_height = X.height + 2 * pH;
    const auto output_width = X.width + 2 * pW;
    VLOG(1) << "Output height: " << output_height;
    VLOG(1) << "Output width:" << output_width;
    VLOG(2) << "Output channels:" << X.featureChannels;
    auto outputWrapper = MPSImageWrapper(
        this,
        &inputWrapper,
        X.numberOfImages,
        output_height,
        output_width,
        X.featureChannels);
    auto commandBuffer = outputWrapper.getCommandBuffer();
    MPSImage* output = outputWrapper.getImage();
    CAFFE_ENFORCE_EQ(output.height, output_height);
    CAFFE_ENFORCE_EQ(output.width, output_width);
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state =
        getMPSCNNContext().getPipelineState(kernelFor(
            output, @"reflection_padding", @"reflection_padding_nonarray"));
    [encoder setComputePipelineState:state];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[output texture] atIndex:1];
    const auto& launchParams =
        spatialPointwiseKernelLaunchParams(state, output);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    inputWrapper.markRead();
    outputWrapper.copyToOutputBlob(Outputs()[0]);

    VLOG(2) << "PadImage took: " << t.MilliSeconds();
    return true;
  }
};

REGISTER_CPU_OPERATOR(MPSCNNPadImage, MPSCNNPadImageOp);
OPERATOR_SCHEMA(MPSCNNPadImage).NumInputs(1).NumOutputs(1);

class MPSCNNMulOp final : public Operator<CPUContext> {
 public:
  MPSCNNMulOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        OperatorBase::GetSingleArgument<int>("broadcast", 0) == 1,
        "MPSCNNMul only supports broadcast");

    OPERATOR_NEEDS_FEATURE(
        OperatorBase::HasArgument("axis") == false,
        "MPSCNNMul does not support axis");
  }

  bool RunOnDevice() override {
    caffe2::Timer t;

    auto wrapper0 = Inputs()[0]->Get<MPSImageWrapper>();
    MPSImage* X0 = wrapper0.getImage();

    const auto& X1 = Input(1);
    CAFFE_ENFORCE_EQ(
        X1.ndim(),
        1,
        "MPSCNNMulOp: Only ndim == 1 for Input(1) is supported for now");

    auto X1_ = [getMPSCNNContext().device
        newBufferWithBytes:X1.template data<float>()
                    length:sizeof(float) * X1.size()
                   options:MTLResourceOptionCPUCacheModeDefault];

    auto outputWrapper = MPSImageWrapper(
        this,
        &wrapper0,
        X0.numberOfImages,
        X0.height,
        X0.width,
        X0.featureChannels);
    auto commandBuffer = outputWrapper.getCommandBuffer();
    MPSImage* output = outputWrapper.getImage();

    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state =
        getMPSCNNContext().getSpecializedPipelineState(
            @"elementwise_mul",
            {{ushort(X0.numberOfImages),
              ushort(X0.featureChannels),
              ushort(X1.dim32(0))}});

    [encoder setComputePipelineState:state];
    [encoder setTexture:[X0 texture] atIndex:0];
    [encoder setBuffer:X1_ offset:0 atIndex:1];
    [encoder setTexture:[output texture] atIndex:2];
    const auto& launchParams =
        spatialPointwiseKernelLaunchParams(state, output);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    wrapper0.markRead();
    outputWrapper.copyToOutputBlob(Outputs()[0]);
    VLOG(2) << "ElementwiseMul took: " << t.MilliSeconds();
    return true;
  }
};

REGISTER_CPU_OPERATOR(MPSCNNMul, MPSCNNMulOp);
OPERATOR_SCHEMA(MPSCNNMul).NumInputs(2).NumOutputs(1).AllowInplace({{0, 0}});

class MPSCNNSubOp final : public Operator<CPUContext> {
 public:
  MPSCNNSubOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        OperatorBase::GetSingleArgument<int>("broadcast", 0) == 1,
        "MPSCNNSub only supports broadcast");

    OPERATOR_NEEDS_FEATURE(
        OperatorBase::HasArgument("axis") == false,
        "MPSCNNSub does not support axis");
  }

  bool RunOnDevice() override {
    caffe2::Timer t;

    auto wrapper0 = Inputs()[0]->Get<MPSImageWrapper>();
    MPSImage* X0 = wrapper0.getImage();

    const auto& X1 = Input(1);
    CAFFE_ENFORCE_EQ(
        X1.ndim(),
        1,
        "MPSCNNSubOp: Only ndim == 1 for Input(1) is supported for now");

    auto X1_ = [getMPSCNNContext().device
        newBufferWithBytes:X1.template data<float>()
                    length:sizeof(float) * X1.size()
                   options:MTLResourceOptionCPUCacheModeDefault];

    auto outputWrapper = MPSImageWrapper(
        this,
        &wrapper0,
        X0.numberOfImages,
        X0.height,
        X0.width,
        X0.featureChannels);
    auto commandBuffer = outputWrapper.getCommandBuffer();
    MPSImage* output = outputWrapper.getImage();

    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state =
        getMPSCNNContext().getSpecializedPipelineState(
            @"elementwise_sub",
            {{ushort(X0.numberOfImages),
              ushort(X0.featureChannels),
              ushort(X1.dim32(0))}});

    [encoder setComputePipelineState:state];
    [encoder setTexture:[X0 texture] atIndex:0];
    [encoder setBuffer:X1_ offset:0 atIndex:1];
    [encoder setTexture:[output texture] atIndex:2];
    const auto& launchParams =
        spatialPointwiseKernelLaunchParams(state, output);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    wrapper0.markRead();
    outputWrapper.copyToOutputBlob(Outputs()[0]);
    VLOG(2) << "ElementwiseSub took: " << t.MilliSeconds();
    return true;
  }
};

REGISTER_CPU_OPERATOR(MPSCNNSub, MPSCNNSubOp);
OPERATOR_SCHEMA(MPSCNNSub).NumInputs(2).NumOutputs(1).AllowInplace({{0, 0}});

class MPSCNNAddOp final : public Operator<CPUContext> {
 public:
  MPSCNNAddOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    caffe2::Timer t;

    auto wrapper0 = Inputs()[0]->Get<MPSImageWrapper>();
    auto wrapper1 = Inputs()[1]->Get<MPSImageWrapper>();
    MPSImage* X0 = wrapper0.getImage();
    MPSImage* X1 = wrapper1.getImage();
    CAFFE_ENFORCE_EQ(wrapper0.getCommandBuffer(), wrapper1.getCommandBuffer());

    auto outputWrapper = MPSImageWrapper(
        this,
        &wrapper0,
        X0.numberOfImages,
        X0.height,
        X0.width,
        X0.featureChannels);
    auto commandBuffer = outputWrapper.getCommandBuffer();
    MPSImage* output = outputWrapper.getImage();
    CAFFE_ENFORCE_EQ(X1.width, X0.width);
    CAFFE_ENFORCE_EQ(X1.height, X0.height);
    CAFFE_ENFORCE_EQ(X1.featureChannels, X0.featureChannels);
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state = getMPSCNNContext().getPipelineState(
        kernelFor(X0, @"elementwise_add", @"elementwise_add_nonarray"));

    [encoder setComputePipelineState:state];
    [encoder setTexture:[X0 texture] atIndex:0];
    [encoder setTexture:[X1 texture] atIndex:1];
    [encoder setTexture:[output texture] atIndex:2];
    const auto& launchParams =
        spatialPointwiseKernelLaunchParams(state, output);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    wrapper0.markRead();
    wrapper1.markRead();
    outputWrapper.copyToOutputBlob(Outputs()[0]);

    VLOG(2) << "ElementwiseAdd took: " << t.MilliSeconds();
    return true;
  }
};

REGISTER_CPU_OPERATOR(MPSCNNAdd, MPSCNNAddOp);
// Not really in-place per-se, but semantically is valid and preserves
// compatibility.
OPERATOR_SCHEMA(MPSCNNAdd).NumInputs(2).NumOutputs(1).AllowInplace({{0, 0}});

class MPSCNNAveragePoolOp final : public ConvPoolOpBase<CPUContext> {
 public:
  MPSCNNAveragePoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NCHW, "Metal only supports NCHW order.");
    OPERATOR_NEEDS_FEATURE(
        kernel_h() == kernel_w(),
        "Metal only supports equal kernel dimension.");
  }

  bool RunOnDeviceWithOrderNCHW() override {
    caffe2::Timer t;
    auto inputWrapper = Inputs()[0]->Get<MPSImageWrapper>();
    MPSImage* X = inputWrapper.getImage();

    if (!pool_ || this->global_pooling_) {
      caffe2::Timer consT;
      this->ComputePads({(int)X.height, (int)X.width});
      pool_ =
          [[MPSCNNPoolingAverage alloc] initWithDevice:getMPSCNNContext().device
                                           kernelWidth:kernel_w()
                                          kernelHeight:kernel_h()
                                       strideInPixelsX:stride_w()
                                       strideInPixelsY:stride_h()];

      [pool_ setEdgeMode:MPSImageEdgeModeClamp];
      MPSOffset offset;
      offset.x = computeMPSAlignOffset(kernel_w(), pad_l());
      offset.y = computeMPSAlignOffset(kernel_h(), pad_t());
      offset.z = 0;
      [pool_ setOffset:offset];
      VLOG(2) << "MPSCNNAveragePool PoolDesc took: " << consT.MilliSeconds();
    }

    CAFFE_ENFORCE_EQ(pool_.strideInPixelsY, stride_h());
    CAFFE_ENFORCE_EQ(pool_.strideInPixelsX, stride_w());
    int output_height;
    int output_width;
    computeOutputHW(this, X.height, X.width, &output_height, &output_width);

    VLOG(2) << "Output height: " << output_height;
    VLOG(2) << "Output width:" << output_width;
    VLOG(2) << "Output channels:" << X.featureChannels;
    auto outputWrapper = MPSImageWrapper(
        this,
        &inputWrapper,
        X.numberOfImages,
        output_height,
        output_width,
        X.featureChannels);
    auto commandBuffer = outputWrapper.getCommandBuffer();
    MPSImage* output = outputWrapper.getImage();
    CAFFE_ENFORCE_EQ(output.height, output_height);
    CAFFE_ENFORCE_EQ(output.width, output_width);
    [pool_ encodeToCommandBuffer:commandBuffer
                     sourceImage:X
                destinationImage:output];
    outputWrapper.copyToOutputBlob(Outputs()[0]);

    VLOG(2) << "MPSCNNAveragePool took: " << t.MilliSeconds();
    return true;
  }

  MPSCNNPoolingAverage* pool_{nullptr};
};

REGISTER_CPU_OPERATOR(MPSCNNAveragePool, MPSCNNAveragePoolOp);
OPERATOR_SCHEMA(MPSCNNAveragePool).NumInputs(1).NumOutputs(1);

class MPSCNNMaxPoolOp final : public ConvPoolOpBase<CPUContext> {
 public:
  MPSCNNMaxPoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NCHW, "Metal only supports NCHW order.");
    OPERATOR_NEEDS_FEATURE(
        kernel_h() == kernel_w(),
        "Metal only supports equal kernel dimension.");
  }

  bool RunOnDeviceWithOrderNCHW() override {
    caffe2::Timer t;
    auto inputWrapper = Inputs()[0]->Get<MPSImageWrapper>();
    MPSImage* X = inputWrapper.getImage();

    if (!pool_ || this->global_pooling_) {
      caffe2::Timer consT;
      this->ComputePads({(int)X.height, (int)X.width});
      pool_ = [[MPSCNNPoolingMax alloc] initWithDevice:getMPSCNNContext().device
                                           kernelWidth:kernel_w()
                                          kernelHeight:kernel_h()
                                       strideInPixelsX:stride_w()
                                       strideInPixelsY:stride_h()];

      [pool_ setEdgeMode:MPSImageEdgeModeClamp];
      MPSOffset offset;
      offset.x = computeMPSAlignOffset(kernel_w(), pad_l());
      offset.y = computeMPSAlignOffset(kernel_h(), pad_t());
      offset.z = 0;
      [pool_ setOffset:offset];
      VLOG(2) << "MPSCNNMaxPool PoolDesc took: " << consT.MilliSeconds();
    }

    CAFFE_ENFORCE_EQ(pool_.strideInPixelsY, stride_h());
    CAFFE_ENFORCE_EQ(pool_.strideInPixelsX, stride_w());

    int output_height;
    int output_width;
    computeOutputHW(this, X.height, X.width, &output_height, &output_width);

    VLOG(2) << "Output height: " << output_height;
    VLOG(2) << "Output width:" << output_width;
    VLOG(2) << "Output channels:" << X.featureChannels;
    auto outputWrapper = MPSImageWrapper(
        this,
        &inputWrapper,
        X.numberOfImages,
        output_height,
        output_width,
        X.featureChannels);
    auto commandBuffer = outputWrapper.getCommandBuffer();
    MPSImage* output = outputWrapper.getImage();
    CAFFE_ENFORCE_EQ(output.height, output_height);
    CAFFE_ENFORCE_EQ(output.width, output_width);
    [pool_ encodeToCommandBuffer:commandBuffer
                     sourceImage:X
                destinationImage:output];
    outputWrapper.copyToOutputBlob(Outputs()[0]);

    VLOG(2) << "MPSCNNMaxPool took: " << t.MilliSeconds();
    return true;
  }

  MPSCNNPoolingMax* pool_{nullptr};
};

REGISTER_CPU_OPERATOR(MPSCNNMaxPool, MPSCNNMaxPoolOp);
OPERATOR_SCHEMA(MPSCNNMaxPool).NumInputs(1).NumOutputs(1);

class MPSCNNSoftmaxOp final : public Operator<CPUContext> {
 public:
  MPSCNNSoftmaxOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    caffe2::Timer t;
    auto inputWrapper = Inputs()[0]->Get<MPSImageWrapper>();
    MPSImage* X = inputWrapper.getImage();
    CAFFE_ENFORCE_EQ(X.height, 1);
    CAFFE_ENFORCE_EQ(X.width, 1);
    if (!softmax_) {
      softmax_ =
          [[MPSCNNSoftMax alloc] initWithDevice:getMPSCNNContext().device];
    }
    auto outputWrapper = MPSImageWrapper(
        this,
        &inputWrapper,
        X.numberOfImages,
        X.height,
        X.width,
        X.featureChannels);
    auto commandBuffer = outputWrapper.getCommandBuffer();
    MPSImage* output = outputWrapper.getImage();
    [softmax_ encodeToCommandBuffer:commandBuffer
                        sourceImage:X
                   destinationImage:output];
    outputWrapper.copyToOutputBlob(Outputs()[0]);
    VLOG(2) << "MPSCNNSoftmax took: " << t.MilliSeconds();
    return true;
  }

  MPSCNNSoftMax* softmax_{nullptr};
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
    auto inputWrapper = Inputs()[0]->template Get<MPSImageWrapper>();
    MPSImage* X = inputWrapper.getImage();
    const auto& W = Input(1);
    const auto& b = Input(2);

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

      MPSCNNConvolutionDescriptor* desc = [MPSCNNConvolutionDescriptor
          cnnConvolutionDescriptorWithKernelWidth:X.width
                                     kernelHeight:X.height
                             inputFeatureChannels:input_channels
                            outputFeatureChannels:output_channels
                                     neuronFilter:Neuron::t()];
      auto data_source = [[ConvDataSource alloc]
          initWithWeight:refilter.data()
                    bias:const_cast<float*>(b.template data<float>())
                    desc:desc];
      fc_ = [[MPSCNNConvolution alloc] initWithDevice:getMPSCNNContext().device
                                              weights:data_source];
    }
    // Note that X.numberOfImages can change between calls, but X.height and
    // X.width are static by definition.
    VLOG(2) << "MPSCNNFC: " << X.numberOfImages << ", " << X.width << ", "
            << X.height << ", " << X.featureChannels << ", " << output_channels;

    [fc_ setClipRect:MTLRegionMake3D(0, 0, 0, 1, 1, X.numberOfImages)];
    MPSOffset off;
    off.x = X.width / 2;
    off.y = X.height / 2;
    off.z = 0;
    [fc_ setOffset:off];
    auto outputWrapper = MPSImageWrapper(
        this, &inputWrapper, X.numberOfImages, 1, 1, output_channels);
    auto commandBuffer = outputWrapper.getCommandBuffer();
    MPSImage* output = outputWrapper.getImage();
    [fc_ encodeToCommandBuffer:commandBuffer
                   sourceImage:X
              destinationImage:output];
    outputWrapper.copyToOutputBlob(Outputs()[0]);
    VLOG(2) << "MPSCNNFC took: " << t.MilliSeconds();
    return true;
  }

  MPSCNNConvolution* fc_{nullptr};
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
    auto inputWrapper = Inputs()[0]->Get<MPSImageWrapper>();
    inputWrapper.copyToOutputBlob(Outputs()[0]);
    return true;
  }
};

REGISTER_CPU_OPERATOR(MPSCNNDropout, MPSCNNDropoutOp);
// Never use the second output (the mask).
OPERATOR_SCHEMA(MPSCNNDropout)
    .NumInputs(1)
    .NumOutputs(1, 2)
    .AllowInplace({{0, 0}});

class MPSCNNConvTransposeOp final : public ConvTransposeUnpoolBase<CPUContext> {
 public:
  MPSCNNConvTransposeOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvTransposeUnpoolBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NCHW, "Metal only supports NCHW order.");
    CAFFE_ENFORCE_EQ(
        kernel_w(), kernel_h(), "Metal only supports equal kernel dimensions");
  }

  bool RunOnDeviceWithOrderNCHW() override {
    caffe2::Timer t;
    auto inputWrapper = Inputs()[0]->Get<MPSImageWrapper>();

    MPSImage* X = inputWrapper.getImage();

    auto& filter = Input(FILTER);
    auto& bias = Input(BIAS);
    CAFFE_ENFORCE(filter.ndim(), 4);
    const int output_channels = filter.dim32(1);
    const int input_channels = filter.dim32(0);

    CAFFE_ENFORCE(X.featureChannels == input_channels, "");
    CAFFE_ENFORCE(filter.dim32(2) == kernel_h(), "");
    CAFFE_ENFORCE(filter.dim32(3) == kernel_w(), "");
    CAFFE_ENFORCE(bias.ndim() == 1, "");
    CAFFE_ENFORCE(bias.dim32(0) == output_channels, "");

    const auto kH = kernel_h();
    const auto kW = kernel_w();

    int output_height =
        (X.height - 1) * stride_h() + kH - pad_b() - pad_t() + adj_h();
    int output_width =
        (X.width - 1) * stride_w() + kW - pad_l() - pad_r() + adj_w();

    VLOG(2) << "Output height: " << output_height;
    VLOG(2) << "Output width:" << output_width;
    VLOG(2) << "Output channels:" << output_channels;

    auto outputWrapper = MPSImageWrapper(
        this,
        &inputWrapper,
        X.numberOfImages,
        output_height,
        output_width,
        output_channels);
    auto commandBuffer = outputWrapper.getCommandBuffer();

    bool runtimeAtLeastIOS11 = SYSTEM_VERSION_GREATER_THAN_OR_EQUAL_TO(@"11.0");
    // initialization
    if (!conv_trans_ && !conv_) {
      caffe2::Timer consT;
      std::vector<float> refilter(kH * kW * output_channels * input_channels);
      refilter.assign(kH * kW * output_channels * input_channels, 0.0f);
      DCHECK_EQ(refilter.size(), filter.size());
      auto* filter_ = filter.template data<float>();
      // For iOS11+ Reformat weights from WT[IC][OC][kH][kW] to
      // W[OC][kH][kW][IC]; For previous versions, reformat weights
      // to W[kH][kW][OC][IC]
      // Also rotate the weight matrix spatially by 180 degrees
      for (auto oc = 0; oc < output_channels; ++oc) {
        for (auto ic = 0; ic < input_channels; ++ic) {
          for (auto kh = 0; kh < kH; ++kh) {
            for (auto kw = 0; kw < kW; ++kw) {
              const auto inputIdx =
                  ic * output_channels * kH * kW + oc * kH * kW + kh * kW + kw;
              int outputIdx;
              if (runtimeAtLeastIOS11) {
                outputIdx = oc * kH * kW * input_channels +
                    (kH - 1 - kh) * kW * input_channels +
                    (kW - 1 - kw) * input_channels + ic;
              } else {
                outputIdx = kh * kW * output_channels * input_channels +
                    kw * output_channels * input_channels +
                    oc * input_channels + ic;
              }
              DCHECK_LT(inputIdx, filter.size());
              DCHECK_LT(outputIdx, filter.size());
              refilter[outputIdx] = filter_[inputIdx];
            }
          }
        }
      }
      DCHECK_EQ(filter.size(), input_channels * output_channels * kH * kW);
      // initialize data structures
      if (runtimeAtLeastIOS11) {
        MPSCNNConvolutionDescriptor* desc = [MPSCNNConvolutionDescriptor
            cnnConvolutionDescriptorWithKernelWidth:kW
                                       kernelHeight:kH
                               inputFeatureChannels:input_channels
                              outputFeatureChannels:output_channels];
        desc.strideInPixelsX = this->stride_w();
        desc.strideInPixelsY = this->stride_h();
        desc.groups = 1;
        auto data_source = [[ConvDataSource alloc]
            initWithWeight:refilter.data()
                      bias:const_cast<float*>(bias.data<float>())
                      desc:desc];

        conv_trans_ = [[MPSCNNConvolutionTranspose alloc]
            initWithDevice:getMPSCNNContext().device
                   weights:data_source];
        MPSOffset offset;
        offset.x = 0;
        offset.y = 0;
        offset.z = 0;
        [conv_trans_ setOffset:offset];
        // kernel offset + padding offset
        conv_trans_.kernelOffsetX = kW / 2 - kW + 1 + this->pad_l();
        conv_trans_.kernelOffsetY = kH / 2 - kH + 1 + this->pad_t();
        VLOG(2) << "MPSCNNConvTranspose ConvDesc took: "
                << consT.MilliSeconds();
      } else {
        MPSCNNConvolutionDescriptor* desc = [MPSCNNConvolutionDescriptor
            cnnConvolutionDescriptorWithKernelWidth:1
                                       kernelHeight:1
                               inputFeatureChannels:input_channels
                              outputFeatureChannels:output_channels * kH * kW
                                       neuronFilter:nil];
        // We need to zero-fill the bias here.
        std::vector<float> fakeBias;
        fakeBias.assign(output_channels * kH * kW, 0);

        desc.strideInPixelsX = 1;
        desc.strideInPixelsY = 1;
        auto data_source =
            [[ConvDataSource alloc] initWithWeight:refilter.data()
                                              bias:fakeBias.data()
                                              desc:desc];
        conv_ =
            [[MPSCNNConvolution alloc] initWithDevice:getMPSCNNContext().device
                                              weights:data_source];
        [conv_ setEdgeMode:MPSImageEdgeModeZero];
        MPSOffset offset;
        offset.x = 0;
        offset.y = 0;
        offset.z = 0;
        [conv_ setOffset:offset];

        const auto biasBytes = divRoundUp(bias.size(), 4) * 4 * 2;
        biasBuffer_ = [getMPSCNNContext().device
            newBufferWithLength:biasBytes
                        options:MTLResourceOptionCPUCacheModeDefault];
        for (auto i = 0; i < bias.size(); ++i) {
          ((float16_t*)[biasBuffer_ contents])[i] = bias.data<float>()[i];
        }

        VLOG(2) << "MPSCNNConvTranspose ConvDesc took: "
                << consT.MilliSeconds();
      } // data structure initialization
    } // initialization
    CAFFE_ENFORCE((conv_trans_ && !conv_) || (!conv_trans_ && conv_));

    // run the computation
    if (conv_trans_) {
      MPSImage* output = outputWrapper.getImage();
      X = inputWrapper.getImage();
      CAFFE_ENFORCE_EQ(conv_trans_.groups, 1);
      [conv_trans_ encodeToCommandBuffer:commandBuffer
                             sourceImage:X
                        destinationImage:output];
    } else {
      CAFFE_ENFORCE_EQ(conv_.strideInPixelsY, 1);
      CAFFE_ENFORCE_EQ(conv_.strideInPixelsX, 1);
      CAFFE_ENFORCE_EQ(conv_.groups, 1);
      CAFFE_ENFORCE_EQ(conv_.inputFeatureChannels, input_channels);
      CAFFE_ENFORCE_EQ(conv_.outputFeatureChannels, output_channels * kH * kW);
      CAFFE_ENFORCE_EQ(conv_.kernelWidth, 1);
      CAFFE_ENFORCE_EQ(conv_.kernelHeight, 1);
      if (divRoundUp(X.numberOfImages * output_channels * kH * kW, 4) >
          kMetalMaxTextureArrLength) {
        LOG(INFO) << "ConvTranspose " << X.numberOfImages << " "
                  << output_channels << " " << kH << " " << kW;
        LOG(ERROR)
            << "arrayLength exceeds the maximum allowed length in texture";
        inputWrapper.cleanup();
        outputWrapper.cleanup();
        return false;
      }
      VLOG(2) << "ConvTranspose:" << output_channels << " " << kH << " " << kW
              << " " << X.numberOfImages;

      auto gemmed = createTemporaryImage(
          this,
          commandBuffer,
          X.numberOfImages,
          X.height,
          X.width,
          output_channels * kH * kW);
      {
        caffe2::Timer gt;
        [conv_ encodeToCommandBuffer:commandBuffer
                         sourceImage:X
                    destinationImage:gemmed];
        VLOG(2) << "MPSCNNConvTranspose GEMM took: " << gt.MilliSeconds();
      }
      MPSImage* output = outputWrapper.getImage();

      {
        caffe2::Timer cit;
        id<MTLComputePipelineState> state =
            getMPSCNNContext().getSpecializedPipelineState(
                @"col2im",
                {{ushort(kernel_h()),
                  ushort(kernel_w()),
                  ushort(stride_h()),
                  ushort(stride_w()),
                  ushort(pad_l()),
                  ushort(pad_t()),
                  ushort(output.featureChannels),
                  ushort(output.numberOfImages),
                  ushort(gemmed.height),
                  ushort(gemmed.width)}});
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:state];
        [encoder setTexture:[gemmed texture] atIndex:0];
        [encoder setTexture:[output texture] atIndex:1];
        [encoder setBuffer:biasBuffer_ offset:0 atIndex:0];
        const auto& launchParams =
            spatialPointwiseKernelLaunchParams(state, output);
        [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
                threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
        [encoder endEncoding];
        gemmed.readCount -= 1;
        VLOG(2) << "MPSCNNConvTranspose upscaling took: " << cit.MilliSeconds();
      }
    }
    outputWrapper.copyToOutputBlob(Outputs()[0]);
    VLOG(2) << "MPSCNNConvTranspose took: " << t.MilliSeconds();
    return true;
  }

  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
  MPSCNNConvolutionTranspose* conv_trans_{nullptr};
  id<MTLBuffer> biasBuffer_;
  MPSCNNConvolution* conv_{nullptr};
};

// No-op init
#define INIT_CONV_TRANSPOSE_NEURON_OP(name)           \
  REGISTER_CPU_OPERATOR(name, MPSCNNConvTransposeOp); \
  OPERATOR_SCHEMA(name).NumInputs(3).NumOutputs(1);

INIT_CONV_TRANSPOSE_NEURON_OP(MPSCNNConvTranspose);
#undef INIT_CONV_TRANSPOSE_NEURON_OP

enum class InstanceNormFusionTy {
  NONE,
  PRELU,
};

template <InstanceNormFusionTy fusionTy>
class MPSCNNInstanceNormOp final : public Operator<CPUContext> {
 public:
  MPSCNNInstanceNormOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    auto inputWrapper = Inputs()[0]->template Get<MPSImageWrapper>();
    MPSImage* X = inputWrapper.getImage();

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
      scaleBuffer_ = [getMPSCNNContext().device
          newBufferWithLength:scaleBytes
                      options:MTLResourceOptionCPUCacheModeDefault];
      biasBuffer_ = [getMPSCNNContext().device
          newBufferWithLength:scaleBytes
                      options:MTLResourceOptionCPUCacheModeDefault];
      for (auto i = 0; i < scale.size(); ++i) {
        ((float16_t*)[scaleBuffer_ contents])[i] =
            scale.template data<float>()[i];
      }
      for (auto i = 0; i < bias.size(); ++i) {
        ((float16_t*)[biasBuffer_ contents])[i] =
            bias.template data<float>()[i];
      }
      if (fusionTy == InstanceNormFusionTy::PRELU) {
        const auto& preluWeight = Input(3);
        preluWeightBuffer_ = [getMPSCNNContext().device
            newBufferWithLength:divRoundUp(preluWeight.size(), 4) * 4 * 2
                        options:MTLResourceOptionCPUCacheModeDefault];
        for (auto i = 0; i < preluWeight.size(); ++i) {
          ((float16_t*)[preluWeightBuffer_ contents])[i] =
              preluWeight.template data<float>()[i];
        }
      }
      VLOG(2) << "Buffer setup took: " << cvt.MilliSeconds();
    }

    auto outputWrapper = MPSImageWrapper(
        this,
        &inputWrapper,
        X.numberOfImages,
        X.height,
        X.width,
        X.featureChannels);
    auto commandBuffer = inputWrapper.getCommandBuffer();
    MPSImage* output = outputWrapper.getImage();

    caffe2::Timer t;
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state =
        getMPSCNNContext().getSpecializedPipelineState(
            kernelFor(X, @"instance_norm", @"instance_norm_nonarray"),
            {{ushort(X.featureChannels),
              fusionTy == InstanceNormFusionTy::PRELU ? ushort(Input(3).size())
                                                      : ushort(0)}});

    [encoder setComputePipelineState:state];
    [encoder setBuffer:scaleBuffer_ offset:0 atIndex:0];
    [encoder setBuffer:biasBuffer_ offset:0 atIndex:1];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[output texture] atIndex:1];
    if (fusionTy == InstanceNormFusionTy::PRELU) {
      [encoder setBuffer:preluWeightBuffer_ offset:0 atIndex:2];
    }
    [encoder dispatchThreadgroups:MTLSizeMake(
                                      1,
                                      1,
                                      X.numberOfImages *
                                          divRoundUp(X.featureChannels, 4))
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [encoder endEncoding];
    inputWrapper.markRead();
    VLOG(2) << "InstanceNorm took: " << t.MilliSeconds();
    outputWrapper.copyToOutputBlob(Outputs()[0]);

    return true;
  }

 private:
  id<MTLBuffer> scaleBuffer_;
  id<MTLBuffer> biasBuffer_;
  id<MTLBuffer> preluWeightBuffer_;
};

REGISTER_CPU_OPERATOR(
    MPSCNNInstanceNorm,
    MPSCNNInstanceNormOp<InstanceNormFusionTy::NONE>);
OPERATOR_SCHEMA(MPSCNNInstanceNorm).NumInputs(3).NumOutputs(1);
REGISTER_CPU_OPERATOR(
    MPSCNNInstanceNormPRelu,
    MPSCNNInstanceNormOp<InstanceNormFusionTy::PRELU>);
OPERATOR_SCHEMA(MPSCNNInstanceNormPRelu).NumInputs(4).NumOutputs(1);

class MPSCNNNormalizePlanarYUVOp final : public Operator<CPUContext> {
 public:
  MPSCNNNormalizePlanarYUVOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    auto inputWrapper = Inputs()[0]->template Get<MPSImageWrapper>();
    MPSImage* X = inputWrapper.getImage();

    const auto& mean = Input(1);
    const auto& std = Input(2);
    CAFFE_ENFORCE_EQ(mean.size(), X.featureChannels);
    CAFFE_ENFORCE_EQ(std.size(), X.featureChannels);
    const auto scaleBytes = divRoundUp(mean.size(), 4) * 4 * 2;
    if (!scaleBuffer_ || !shiftBuffer_ || scaleBuffer_.length != scaleBytes ||
        shiftBuffer_.length != scaleBytes) {
      caffe2::Timer cvt;
      scaleBuffer_ = [getMPSCNNContext().device
          newBufferWithLength:scaleBytes
                      options:MTLResourceOptionCPUCacheModeDefault];
      shiftBuffer_ = [getMPSCNNContext().device
          newBufferWithLength:scaleBytes
                      options:MTLResourceOptionCPUCacheModeDefault];
      // op computes (X - mean) / std = X * 1/std + (-mean/std)
      // Thus set scale = 1.0/std, shift = (-mean/std)
      for (auto i = 0; i < mean.size(); ++i) {
        ((float16_t*)[scaleBuffer_ contents])[i] =
            1.0 / double(std.template data<float>()[i]);
        ((float16_t*)[shiftBuffer_ contents])[i] =
            double(-mean.template data<float>()[i]) /
            double(std.template data<float>()[i]);
      }
      VLOG(2) << "Buffer setup took: " << cvt.MilliSeconds();
    }

    auto outputWrapper = MPSImageWrapper(
        this,
        &inputWrapper,
        X.numberOfImages,
        X.height,
        X.width,
        X.featureChannels);
    auto commandBuffer = inputWrapper.getCommandBuffer();
    MPSImage* output = outputWrapper.getImage();

    caffe2::Timer t;
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state =
        getMPSCNNContext().getSpecializedPipelineState(
            kernelFor(X, @"affine", @"affine_nonarray"),
            {ushort(X.featureChannels)});

    [encoder setComputePipelineState:state];
    [encoder setBuffer:scaleBuffer_ offset:0 atIndex:0];
    [encoder setBuffer:shiftBuffer_ offset:0 atIndex:1];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[output texture] atIndex:1];
    const auto& launchParams =
        spatialPointwiseKernelLaunchParams(state, output);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    inputWrapper.markRead();
    VLOG(2) << "InstanceNorm took: " << t.MilliSeconds();
    outputWrapper.copyToOutputBlob(Outputs()[0]);

    return true;
  }

 private:
  id<MTLBuffer> scaleBuffer_;
  id<MTLBuffer> shiftBuffer_;
};

REGISTER_CPU_OPERATOR(MPSCNNNormalizePlanarYUV, MPSCNNNormalizePlanarYUVOp);
OPERATOR_SCHEMA(MPSCNNNormalizePlanarYUV).NumInputs(3).NumOutputs(1);

class MPSCNNPReluOp final : public Operator<CPUContext> {
 public:
  MPSCNNPReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    auto inputWrapper = Inputs()[0]->Get<MPSImageWrapper>();
    const MPSImage* X = inputWrapper.getImage();

    const auto& scale = Input(1);
    const auto scaleBytes = divRoundUp(scale.size(), 4) * 4 * 2;
    if (!scaleBuffer_ || scaleBuffer_.length != scaleBytes) {
      caffe2::Timer cvt;
      scaleBuffer_ = [getMPSCNNContext().device
          newBufferWithLength:scaleBytes
                      options:MTLResourceOptionCPUCacheModeDefault];
      for (auto i = 0; i < scale.size(); ++i) {
        ((float16_t*)[scaleBuffer_ contents])[i] = scale.data<float>()[i];
      }
      VLOG(2) << "Buffer setup took: " << cvt.MilliSeconds();
    }

    auto outputWrapper = MPSImageWrapper(
        this,
        &inputWrapper,
        X.numberOfImages,
        X.height,
        X.width,
        X.featureChannels);
    auto commandBuffer = inputWrapper.getCommandBuffer();
    MPSImage* output = outputWrapper.getImage();
    caffe2::Timer t;
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state =
        getMPSCNNContext().getSpecializedPipelineState(
            kernelFor(X, @"prelu_nonshared", @"prelu_nonshared_nonarray"),
            {{ushort(X.featureChannels), ushort(scale.size())}});

    [encoder setComputePipelineState:state];
    [encoder setBuffer:scaleBuffer_ offset:0 atIndex:0];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[output texture] atIndex:1];

    const auto& launchParams =
        spatialPointwiseKernelLaunchParams(state, output);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    inputWrapper.markRead();
    VLOG(2) << "PRelu took: " << t.MilliSeconds();
    outputWrapper.copyToOutputBlob(Outputs()[0]);

    return true;
  }

 private:
  id<MTLBuffer> scaleBuffer_;
};

REGISTER_CPU_OPERATOR(MPSCNNPRelu, MPSCNNPReluOp);
// Allow in-place isn't *really* valid here, since nothing is in-place for Metal
// texture arrays, but requires re-export.
OPERATOR_SCHEMA(MPSCNNPRelu).NumInputs(2).NumOutputs(1).AllowInplace({{0, 0}});

class MPSCNNRoIWarpOp final : public Operator<CPUContext> {
 public:
  MPSCNNRoIWarpOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        spatial_scale_(
            OperatorBase::GetSingleArgument<float>("spatial_scale", 1.)),
        pooled_height_(OperatorBase::GetSingleArgument<int>("pooled_h", 1)),
        pooled_width_(OperatorBase::GetSingleArgument<int>("pooled_w", 1)),
        sampling_ratio_(
            OperatorBase::GetSingleArgument<int>("sampling_ratio", -1)) {
    CAFFE_ENFORCE_GT(spatial_scale_, 0);
    CAFFE_ENFORCE_GT(pooled_height_, 0);
    CAFFE_ENFORCE_GT(pooled_width_, 0);
    CAFFE_ENFORCE_GE(sampling_ratio_, 0);
    VLOG(1) << "spatial_scale: " << spatial_scale_;
    VLOG(1) << "pooled_h: " << pooled_height_;
    VLOG(1) << "pooled_w: " << pooled_width_;
    VLOG(1) << "sampling_ratio: " << sampling_ratio_;
  }

  bool RunOnDevice() override {
    caffe2::Timer t;
    auto inputWrapper = Inputs()[0]->Get<MPSImageWrapper>();
    auto X = inputWrapper.getImage();
    CAFFE_ENFORCE_EQ(X.numberOfImages, 1);
    const auto& R = Input(1);
    CAFFE_ENFORCE_EQ(R.ndim(), 2);
    CAFFE_ENFORCE(R.dim32(1) == 4 || R.dim32(1) == 5);
    const auto roiBytes = R.dim32(0) * 4 * sizeof(float16_t);
    if (!roiBuffer_ || roiBuffer_.length != roiBytes) {
      caffe2::Timer cvt;
      roiBuffer_ = [getMPSCNNContext().device
          newBufferWithLength:roiBytes
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
    auto featureChannels = X.featureChannels;
    VLOG(1) << "RoIWarp input size:" << X.numberOfImages << " "
            << featureChannels << " " << X.height << " " << X.width;
    VLOG(1) << "RoIWarp output size:" << R.dim32(0) << " " << featureChannels
            << " " << pooled_width_ << " " << pooled_height_;
    if (R.dim32(0) <= 0) {
      LOG(ERROR) << "number of RoIs <= 0 in RoIWarp " << R.dim32(0);
      inputWrapper.cleanup();
      return false;
    }
    if (divRoundUp(R.dim32(0) * featureChannels, 4) >
        kMetalMaxTextureArrLength) {
      LOG(INFO) << "MPSCNNRoIWarp " << R.dim32(0) << " " << featureChannels;
      LOG(ERROR) << "arrayLength exceeds the maximum allowed length in texture";
      inputWrapper.cleanup();
      return false;
    }
    auto outputWrapper = MPSImageWrapper(
        this,
        &inputWrapper,
        R.dim32(0),
        pooled_height_,
        pooled_width_,
        featureChannels);
    auto commandBuffer = outputWrapper.getCommandBuffer();
    MPSImage* output = outputWrapper.getImage();
    VLOG(1) << "output: " << output.numberOfImages << ", "
            << output.featureChannels << ", " << output.height << ", "
            << output.width;
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state =
        getMPSCNNContext().getSpecializedPipelineState(
            @"roi_warp",
            {{ushort(spatial_scale_ * 10000),
              ushort(sampling_ratio_),
              ushort(featureChannels),
              ushort(X.numberOfImages),
              ushort(output.numberOfImages)}});

    [encoder setComputePipelineState:state];
    [encoder setBuffer:roiBuffer_ offset:0 atIndex:0];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[output texture] atIndex:1];

    const auto& launchParams =
        spatialPointwiseKernelLaunchParams(state, output);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    inputWrapper.markRead();
    VLOG(2) << "RoIWarp took: " << t.MilliSeconds();
    VLOG(1) << "ROIWarp size: " << output.numberOfImages << ", "
            << output.featureChannels << ", " << output.height << ", "
            << output.width;
    outputWrapper.copyToOutputBlob(Outputs()[0]);

    return true;
  }

 private:
  float spatial_scale_;
  int pooled_height_;
  int pooled_width_;
  int sampling_ratio_;

  id<MTLBuffer> roiBuffer_;
};

REGISTER_CPU_OPERATOR(MPSCNNRoIWarp, MPSCNNRoIWarpOp);
OPERATOR_SCHEMA(MPSCNNRoIWarp).NumInputs(2).NumOutputs(1);

class MPSCNNGenerateProposalsCPPOp final : public Operator<CPUContext> {
 public:
  MPSCNNGenerateProposalsCPPOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        spatial_scale_(
            OperatorBase::GetSingleArgument<float>("spatial_scale", 1.0 / 16)),
        feat_stride_(1.0 / spatial_scale_),
        rpn_pre_nms_topN_(
            OperatorBase::GetSingleArgument<int>("pre_nms_topN", 6000)),
        rpn_post_nms_topN_(
            OperatorBase::GetSingleArgument<int>("post_nms_topN", 300)),
        rpn_nms_thresh_(
            OperatorBase::GetSingleArgument<float>("nms_thresh", 0.7f)),
        rpn_min_size_(OperatorBase::GetSingleArgument<float>("min_size", 16)) {}

  template <class Derived1, class Derived2>
  std::vector<int> nms_metal(
      const Eigen::ArrayBase<Derived1>& proposals, // EArrXXf
      const Eigen::ArrayBase<Derived2>& scores, // EArrXf
      const std::vector<int>& sorted_indices,
      float thresh) const {
    CAFFE_ENFORCE_EQ(proposals.rows(), scores.rows());
    CAFFE_ENFORCE_EQ(proposals.cols(), 4);
    CAFFE_ENFORCE_EQ(scores.cols(), 1);
    CAFFE_ENFORCE_LE(sorted_indices.size(), proposals.rows());

    std::vector<float> proposals_cpu(proposals.size());
    Eigen::Map<ERArrXXf>(
        &proposals_cpu[0], proposals.rows(), proposals.cols()) = proposals;

    int box_num = sorted_indices.size();
    int col_blocks = divRoundUp(box_num, maxThreadsPerThreadgroup);
    auto pre_nms_size = box_num;
    auto preNmsProposalsBuffer_ = [getMPSCNNContext().device
        newBufferWithBytes:proposals_cpu.data()
                    length:proposals.size() * sizeof(float)
                   options:MTLResourceOptionCPUCacheModeDefault];
    auto sortedIndicesBuffer_ = [getMPSCNNContext().device
        newBufferWithBytes:sorted_indices.data()
                    length:pre_nms_size * sizeof(int)
                   options:MTLResourceOptionCPUCacheModeDefault];

    int pose_nms_size = fmin(rpn_post_nms_topN_, pre_nms_size);
    // round pose_nms_size up to the next power of 2
    int batch_size = pow(2, ceil(log(pose_nms_size) / log(2)));

    auto maskBuffer_ = [getMPSCNNContext().device
        newBufferWithLength:batch_size * col_blocks * sizeof(uint32_t)
                    options:MTLResourceOptionCPUCacheModeDefault];
    std::vector<uint32_t> masks(batch_size * col_blocks);

    std::vector<int> keep(pose_nms_size);
    int num_to_keep = 0;
    bool terminate = false;
    std::vector<uint32_t> remv(col_blocks);

    for (int offset = 0; !terminate && offset < box_num; offset += batch_size) {
      auto commandBuffer = [getMPSCNNContext().commandQueue commandBuffer];
      auto encoder = [commandBuffer computeCommandEncoder];
      auto state = getMPSCNNContext().getSpecializedPipelineState(
          @"nms",
          {{ushort(batch_size),
            maxThreadsPerThreadgroup,
            ushort(rpn_nms_thresh_ * 10000),
            ushort(offset)}});
      [encoder setComputePipelineState:state];
      [encoder setBuffer:maskBuffer_ offset:0 atIndex:0];
      [encoder setBuffer:preNmsProposalsBuffer_ offset:0 atIndex:1];
      [encoder setBuffer:sortedIndicesBuffer_ offset:0 atIndex:2];
      const auto threadsPerThreadgroup =
          MTLSizeMake(maxThreadsPerThreadgroup, 1, 1);
      const auto threadgroupsPerGrid = MTLSizeMake(
          divRoundUp(batch_size, maxThreadsPerThreadgroup),
          divRoundUp(box_num, maxThreadsPerThreadgroup),
          1);
      [encoder dispatchThreadgroups:threadgroupsPerGrid
              threadsPerThreadgroup:threadsPerThreadgroup];
      [encoder endEncoding];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
      uint32_t* maskBufferPointer = (uint32_t*)[maskBuffer_ contents];
      std::copy(
          maskBufferPointer,
          maskBufferPointer + (maskBuffer_.length / sizeof(uint32_t)),
          masks.begin());

      for (int i = offset; i < fmin(offset + batch_size, box_num); ++i) {
        int nblock = i / maxThreadsPerThreadgroup;
        int inblock = i % maxThreadsPerThreadgroup;
        if (!(remv[nblock] & (1U << inblock))) {
          keep[num_to_keep++] = sorted_indices[i];
          if (num_to_keep >= pose_nms_size) {
            terminate = true;
            break;
          }
          uint* p = &masks[0] + (i - offset) * col_blocks;
          for (int j = nblock; j < col_blocks; j++) {
            remv[j] |= p[j];
          }
        }
      }
    }
    keep.resize(num_to_keep);
    return keep;
  }
  void ProposalsForOneImage(
      const Eigen::Array3f& im_info,
      const Eigen::Map<const ERMatXf>& all_anchors,
      const utils::ConstTensorView<float>& bbox_deltas_tensor,
      const utils::ConstTensorView<float>& scores_tensor,
      ERArrXXf* out_boxes,
      EArrXf* out_probs) const {
    const auto& pre_nms_topN = rpn_pre_nms_topN_;
    const auto& post_nms_topN = rpn_post_nms_topN_;
    const auto& nms_thresh = rpn_nms_thresh_;
    const auto& min_size = rpn_min_size_;

    // Transpose and reshape predicted bbox transformations to get them
    // into the same order as the anchors:
    //   - bbox deltas will be (4 * A, H, W) format from conv output
    //   - transpose to (H, W, 4 * A)
    //   - reshape to (H * W * A, 4) where rows are ordered by (H, W, A)
    //     in slowest to fastest order to match the enumerated anchors
    CAFFE_ENFORCE_EQ(bbox_deltas_tensor.ndim(), 3);
    CAFFE_ENFORCE_EQ(bbox_deltas_tensor.dim(0) % 4, 0);
    auto A = bbox_deltas_tensor.dim(0) / 4;
    auto H = bbox_deltas_tensor.dim(1);
    auto W = bbox_deltas_tensor.dim(2);
    // equivalent to python code
    //  bbox_deltas = bbox_deltas.transpose((1, 2, 0)).reshape((-1, 4))
    ERArrXXf bbox_deltas(H * W * A, 4);
    Eigen::Map<ERMatXf>(bbox_deltas.data(), H * W, 4 * A) =
        Eigen::Map<const ERMatXf>(bbox_deltas_tensor.data(), A * 4, H * W)
            .transpose();
    CAFFE_ENFORCE_EQ(bbox_deltas.rows(), all_anchors.rows());

    // - scores are (A, H, W) format from conv output
    // - transpose to (H, W, A)
    // - reshape to (H * W * A, 1) where rows are ordered by (H, W, A)
    //   to match the order of anchors and bbox_deltas
    CAFFE_ENFORCE_EQ(scores_tensor.ndim(), 3);
    CAFFE_ENFORCE_EQ(scores_tensor.dims(), (vector<int>{A, H, W}));
    // equivalent to python code
    // scores = scores.transpose((1, 2, 0)).reshape((-1, 1))
    EArrXf scores(scores_tensor.size());
    Eigen::Map<ERMatXf>(scores.data(), H * W, A) =
        Eigen::Map<const ERMatXf>(scores_tensor.data(), A, H * W).transpose();
    // Transform anchors into proposals via bbox transformations
    auto proposals = utils::bbox_transform(all_anchors.array(), bbox_deltas);

    // 2. clip proposals to image (may result in proposals with zero area
    // that will be removed in the next step)
    proposals = utils::clip_boxes(proposals, im_info[0], im_info[1]);

    // 3. remove predicted boxes with either height or width < min_size
    auto keep = utils::filter_boxes(proposals, min_size, im_info);

    DCHECK_LE(keep.size(), scores.size());

    // 4. sort all (proposal, score) pairs by score from highest to lowest
    // 5. take top pre_nms_topN (e.g. 6000)
    std::sort(keep.begin(), keep.end(), [&scores](int lhs, int rhs) {
      return scores[lhs] > scores[rhs];
    });

    if (pre_nms_topN > 0 && pre_nms_topN < keep.size()) {
      keep.resize(pre_nms_topN);
    }

    // 6. apply loose nms (e.g. threshold = 0.7)
    // 7. take after_nms_topN (e.g. 300)
    // 8. return the top proposals (-> RoIs top)
    keep = nms_metal(proposals, scores, keep, nms_thresh);
    if (post_nms_topN > 0 && post_nms_topN < keep.size()) {
      keep.resize(post_nms_topN);
    }
    // Generate outputs
    utils::GetSubArrayRows(proposals, utils::AsEArrXt(keep), out_boxes);
    utils::GetSubArray(scores, utils::AsEArrXt(keep), out_probs);
  }

  bool RunOnDevice() override {
    const auto& scores = Input(0);
    const auto& bbox_deltas = Input(1);
    const auto& im_info_tensor = Input(2);
    const auto& anchors = Input(3);
    auto* out_rois = Output(0);
    auto* out_rois_probs = Output(1);

    CAFFE_ENFORCE_EQ(scores.ndim(), 4, scores.ndim());
    CAFFE_ENFORCE(scores.template IsType<float>(), scores.meta().name());
    const auto num_images = scores.dim(0);
    const auto A = scores.dim(1);
    const auto height = scores.dim(2);
    const auto width = scores.dim(3);
    const auto K = height * width;

    // bbox_deltas: (num_images, A * 4, H, W)
    CAFFE_ENFORCE_EQ(
        bbox_deltas.dims(), (vector<TIndex>{num_images, 4 * A, height, width}));

    // im_info_tensor: (num_images, 3), format [height, width, scale; ...]
    CAFFE_ENFORCE_EQ(im_info_tensor.dims(), (vector<TIndex>{num_images, 3}));
    CAFFE_ENFORCE(
        im_info_tensor.template IsType<float>(), im_info_tensor.meta().name());

    // anchors: (A, 4)
    CAFFE_ENFORCE_EQ(anchors.dims(), (vector<TIndex>{A, 4}));
    CAFFE_ENFORCE(anchors.template IsType<float>(), anchors.meta().name());
    // Broadcast the anchors to all pixels
    auto all_anchors_vec =
        utils::ComputeAllAnchors(anchors, height, width, feat_stride_);
    Eigen::Map<const ERMatXf> all_anchors(all_anchors_vec.data(), K * A, 4);

    Eigen::Map<const ERArrXXf> im_info(
        im_info_tensor.data<float>(),
        im_info_tensor.dim(0),
        im_info_tensor.dim(1));

    const int roi_col_count = 5;
    out_rois->Resize(0, roi_col_count);
    out_rois_probs->Resize(0);
    Timer t1;
    // Use openmp for acceleration?
    for (int i = 0; i < num_images; i++) {
      auto cur_im_info = im_info.row(i);
      auto cur_bbox_deltas = GetSubTensorView<float>(bbox_deltas, i);
      auto cur_scores = GetSubTensorView<float>(scores, i);

      ERArrXXf im_i_boxes;
      EArrXf im_i_probs;
      ProposalsForOneImage(
          cur_im_info,
          all_anchors,
          cur_bbox_deltas,
          cur_scores,
          &im_i_boxes,
          &im_i_probs);

      int csz = im_i_boxes.rows();
      int cur_start_idx = out_rois->dim(0);

      out_rois->Extend(csz, 50, &context_);
      out_rois_probs->Extend(csz, 50, &context_);

      // write rois
      Eigen::Map<ERArrXXf> cur_rois(
          out_rois->mutable_data<float>() + cur_start_idx * roi_col_count,
          csz,
          5);
      cur_rois.col(0).setConstant(i);
      cur_rois.block(0, 1, csz, 4) = im_i_boxes;

      // write rois_probs
      Eigen::Map<EArrXf>(
          out_rois_probs->mutable_data<float>() + cur_start_idx, csz) =
          im_i_probs;
    }

    return true;
  }

 protected:
  // spatial_scale_ must be declared before feat_stride_
  float spatial_scale_{1.0};
  float feat_stride_{1.0};

  // RPN_PRE_NMS_TOP_N
  ushort rpn_pre_nms_topN_{6000};
  // RPN_POST_NMS_TOP_N
  ushort rpn_post_nms_topN_{300};
  // RPN_NMS_THRESH
  float rpn_nms_thresh_{0.7};
  // RPN_MIN_SIZE
  float rpn_min_size_{16};
  // threads per thread group, used in nms
  ushort maxThreadsPerThreadgroup{32};

 private:
  id<MTLBuffer> out_rois_{nullptr};
  id<MTLBuffer> out_rois_probs_{nullptr};
};

REGISTER_CPU_OPERATOR(MPSCNNGenerateProposalsCPP, MPSCNNGenerateProposalsCPPOp);
OPERATOR_SCHEMA(MPSCNNGenerateProposalsCPP).NumInputs(4).NumOutputs(2);

class MPSCNNSpatialBNOp final : public SpatialBNOp<CPUContext> {
 public:
  MPSCNNSpatialBNOp(const OperatorDef& operator_def, Workspace* ws)
      : SpatialBNOp<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    auto inputWrapper = Inputs()[0]->Get<MPSImageWrapper>();
    const MPSImage* X = inputWrapper.getImage();
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
      scaleBuffer_ = [getMPSCNNContext().device
          newBufferWithLength:scaleBytes
                      options:MTLResourceOptionCPUCacheModeDefault];
      shiftBuffer_ = [getMPSCNNContext().device
          newBufferWithLength:scaleBytes
                      options:MTLResourceOptionCPUCacheModeDefault];
      for (auto i = 0; i < scale.size(); ++i) {
        // We can fuse the output computation as follows:
        //   ((x - est_mean) * (inv_var) * scale + bias
        // to
        //   (x * inv_var * scale) + (bias - est_mean * inv_var * scale)

        const auto inv_std = 1.0 / std::sqrt(var.data<float>()[i] + epsilon_);
        ((float16_t*)[scaleBuffer_ contents])[i] =
            scale.data<float>()[i] * inv_std;
        ((float16_t*)[shiftBuffer_ contents])[i] = bias.data<float>()[i] -
            mean.data<float>()[i] * inv_std * scale.data<float>()[i];
      }
      VLOG(2) << "Buffer setup took: " << cvt.MilliSeconds();
    }

    auto outputWrapper = MPSImageWrapper(
        this,
        &inputWrapper,
        X.numberOfImages,
        X.height,
        X.width,
        X.featureChannels);
    auto commandBuffer = outputWrapper.getCommandBuffer();
    MPSImage* output = outputWrapper.getImage();
    caffe2::Timer t;
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state =
        getMPSCNNContext().getSpecializedPipelineState(
            kernelFor(output, @"affine", @"affine_nonarray"),
            {ushort(X.featureChannels)});

    [encoder setComputePipelineState:state];
    [encoder setBuffer:scaleBuffer_ offset:0 atIndex:0];
    [encoder setBuffer:shiftBuffer_ offset:0 atIndex:1];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[output texture] atIndex:1];

    const auto& launchParams =
        spatialPointwiseKernelLaunchParams(state, output);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    inputWrapper.markRead();
    VLOG(2) << "SpatialBN took: " << t.MilliSeconds();
    outputWrapper.copyToOutputBlob(Outputs()[0]);

    return true;
  }

 private:
  id<MTLBuffer> scaleBuffer_;
  id<MTLBuffer> shiftBuffer_;
};

REGISTER_CPU_OPERATOR(MPSCNNSpatialBN, MPSCNNSpatialBNOp);
OPERATOR_SCHEMA(MPSCNNSpatialBN).NumInputs(5).NumOutputs(1);

class MPSCNNConcatOp final : public Operator<CPUContext> {
 public:
  MPSCNNConcatOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {
    // Only handle three inputs for now.
    OPERATOR_NEEDS_FEATURE(
        Inputs().size() <= 4, "MPSCNNConcat only handles up to four inputs");
  }

  bool RunOnDevice() override {
    auto Wrapper = [&](size_t i) {
      return Inputs()[i]->template Get<MPSImageWrapper>();
    };
    auto cb = [&](size_t i) { return Wrapper(i).getCommandBuffer(); };
    auto X = [&](size_t i) { return Wrapper(i).getImage(); };

    // C0, C1, C2, C3, C, N
    std::vector<ushort> channels = {
        {0, 0, 0, 0, 0, ushort(X(0).numberOfImages)}};
    size_t channelCount = 0;
    for (auto i = 0; i < Inputs().size(); ++i) {
      // this does not hold for non-temp images inputs
      CAFFE_ENFORCE_EQ(cb(0), cb(i));
      CAFFE_ENFORCE_EQ(X(0).height, X(i).height);
      CAFFE_ENFORCE_EQ(X(0).width, X(i).width);
      channels[i] = X(i).featureChannels;
      channelCount += X(i).featureChannels;
    }
    channels[4] = channelCount;

    auto wrapper0 = Inputs()[0]->template Get<MPSImageWrapper>();
    auto outputWrapper = MPSImageWrapper(
        this,
        &wrapper0,
        X(0).numberOfImages,
        X(0).height,
        X(0).width,
        channelCount);
    auto commandBuffer = outputWrapper.getCommandBuffer();
    MPSImage* output = outputWrapper.getImage();
    caffe2::Timer t;
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> state =
        getMPSCNNContext().getSpecializedPipelineState(@"concat", channels);

    [encoder setComputePipelineState:state];
    for (auto i = 0; i < Inputs().size(); ++i) {
      [encoder setTexture:[X(i) texture] atIndex:i];
    }
    [encoder setTexture:[output texture] atIndex:5];
    const auto& launchParams =
        spatialPointwiseKernelLaunchParams(state, output);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    for (auto i = 0; i < Inputs().size(); ++i) {
      Wrapper(i).markRead();
    }

    VLOG(2) << "Concat took: " << t.MilliSeconds();
    outputWrapper.copyToOutputBlob(Outputs()[0]);
    return true;
  }
};

REGISTER_CPU_OPERATOR(MPSCNNConcat, MPSCNNConcatOp);
// Only store one output in practice (ignore the shape argument).
OPERATOR_SCHEMA(MPSCNNConcat).NumInputs(2, 4).NumOutputs(1, 2);

class MPSCNNResizeNearestOp final : public Operator<CPUContext> {
 public:
  MPSCNNResizeNearestOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {
    width_scale_ = OperatorBase::GetSingleArgument<float>("width_scale", 1);
    height_scale_ = OperatorBase::GetSingleArgument<float>("height_scale", 1);
    CAFFE_ENFORCE_GT(width_scale_, 0);
    CAFFE_ENFORCE_GT(height_scale_, 0);

    // due to the way we pass these parameters, we don't support the scale to be
    // larger than 6.5
    CAFFE_ENFORCE_LE(width_scale_, 6.5);
    CAFFE_ENFORCE_LE(height_scale_, 6.5);
  }

  bool RunOnDevice() override {
    auto inputWrapper = Inputs()[0]->Get<MPSImageWrapper>();
    const MPSImage* X = inputWrapper.getImage();

    const int N = X.numberOfImages, C = X.featureChannels, H = X.height,
              W = X.width;
    int output_width = W * width_scale_;
    int output_height = H * height_scale_;
    auto outputWrapper =
        MPSImageWrapper(this, &inputWrapper, N, output_height, output_width, C);
    auto commandBuffer = inputWrapper.getCommandBuffer();
    MPSImage* output = outputWrapper.getImage();

    auto encoder = [commandBuffer computeCommandEncoder];
    auto state = getMPSCNNContext().getSpecializedPipelineState(
        kernelFor(output, @"resize_nearest", @"resize_nearest_nonarray"),
        {{ushort(output_height),
          ushort(output_width),
          ushort(height_scale_ * 10000),
          ushort(width_scale_ * 10000)}});
    [encoder setComputePipelineState:state];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[output texture] atIndex:1];
    auto launchParams = spatialPointwiseKernelLaunchParams(state, output);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    inputWrapper.markRead();
    outputWrapper.copyToOutputBlob(Outputs()[0]);

    return true;
  }

 protected:
  float width_scale_;
  float height_scale_;
};

REGISTER_CPU_OPERATOR(MPSCNNResizeNearest, MPSCNNResizeNearestOp);
OPERATOR_SCHEMA(MPSCNNResizeNearest).NumInputs(1).NumOutputs(1);
}

CAFFE_KNOWN_TYPE(MPSImageWrapper);
} // namespace caffe2

#endif
