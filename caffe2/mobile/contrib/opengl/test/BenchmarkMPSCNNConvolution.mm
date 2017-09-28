/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "caffe2/core/flags.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/predictor.h"
#include "caffe2/core/timer.h"

#import "caffe2/mobile/contrib/ios/metal/MetalContext.h"

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

static void testMPSCNN(int input_width,
                       int input_height,
                       int kernel_size,
                       int input_channels,
                       int output_channels,
                       int iterations) {
  MetalContext* metalContext = [MetalContext getContext];

  if (metalContext) {
    int pad = 0;
    int stride = 1;

    int output_height = (input_height - kernel_size + 2 * pad) / stride + 1;
    int output_width = (input_width - kernel_size + 2 * pad) / stride + 1;

    float* kernel_data = (float*)malloc(kernel_size * kernel_size * input_channels *
                                        output_channels * sizeof(float));
    float* bias_data = (float*)malloc(output_channels * sizeof(float));

    MPSCNNConvolution* conv = [[MPSCNNConvolution alloc]
               initWithDevice:metalContext.device
        convolutionDescriptor:[MPSCNNConvolutionDescriptor
                                  cnnConvolutionDescriptorWithKernelWidth:kernel_size
                                                             kernelHeight:kernel_size
                                                     inputFeatureChannels:input_channels
                                                    outputFeatureChannels:output_channels
                                                             neuronFilter:NULL]
                kernelWeights:kernel_data
                    biasTerms:bias_data
                        flags:MPSCNNConvolutionFlagsNone];
    [conv setEdgeMode:MPSImageEdgeModeClamp];

    MPSImage* input_image = [[MPSImage alloc]
         initWithDevice:metalContext.device
        imageDescriptor:[MPSImageDescriptor
                            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                                       width:input_width
                                                      height:input_height
                                             featureChannels:input_channels]];

    MPSImage* output_image = [[MPSImage alloc]
         initWithDevice:metalContext.device
        imageDescriptor:[MPSImageDescriptor
                            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                                       width:output_width
                                                      height:output_height
                                             featureChannels:output_channels]];

    dispatch_semaphore_t gpu_execution_done = dispatch_semaphore_create(0);

    for (int i = 0; i < iterations; i++) {
      id<MTLCommandBuffer> commandBuffer = [metalContext.commandQueue commandBuffer];

      [conv encodeToCommandBuffer:commandBuffer
                      sourceImage:input_image
                 destinationImage:output_image];

      [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> commandBuffer) {
        if (i == iterations - 1)
          dispatch_semaphore_signal(gpu_execution_done);
      }];

      [commandBuffer commit];
    }

    dispatch_semaphore_wait(gpu_execution_done, DISPATCH_TIME_FOREVER);
  } else {
    printf("No Metal context for MPSCNNConvolution\n");
  }
}

void BenchMPSCNN(
    int inputC, int outputC, int kW, int kH, int stride, int inW, int inH, bool transposed) {
  testMPSCNN(inW, inH, kW, inputC, outputC, 10);

  caffe2::Timer t;
  auto niter = 50;

  testMPSCNN(inW, inH, kW, inputC, outputC, 50);

  printf("%s(%d -> %d, %dx%d - %dx%d - %s) took: %.4f ms/iter\n",
         "Conv",
         inputC,
         outputC,
         inW,
         inH,
         kW,
         kH,
         "MPSCNN",
         t.MilliSeconds() / (float)niter);
}

#include "/ios/metal/FBMetalCNNConvolution.h"

static void testMetalConvolution(int input_width,
                                 int input_height,
                                 int kernel_size,
                                 int input_channels,
                                 int output_channels,
                                 int iterations) {
  MetalContext* metalContext = [MetalContext getContext];

  int pad = 0;
  int stride = 1;

  int output_height = (input_height - kernel_size + 2 * pad) / stride + 1;
  int output_width = (input_width - kernel_size + 2 * pad) / stride + 1;

  int kernels_per_convolution = output_channels;

  while (kernels_per_convolution > MAX_KERNELS_PER_CONVOLUTION) {
    if (kernels_per_convolution % 3 == 0)
      kernels_per_convolution /= 3;
    else if (kernels_per_convolution % 2 == 0)
      kernels_per_convolution /= 2;
    else {
      printf("The number of output channels must be a multiple of 2 or 3\n");
    }
  }

  const int kernel_stride = kernels_per_convolution;

  // clang-format on

  const int convolutions = output_channels / kernels_per_convolution;

  FBMetalCNNConstantValues constantValues = FBMetalCNNConstantValues(output_width,
                                                                     output_height,
                                                                     input_width + 2 * pad,
                                                                     input_height + 2 * pad,
                                                                     stride,
                                                                     stride,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     kernel_size,
                                                                     kernel_size,
                                                                     input_channels,
                                                                     kernels_per_convolution,
                                                                     kernels_per_convolution);

  FBMetalCNNConvolution* convolution = [FBMetalCNNConvolution filterWithContext:metalContext
                                                                       channels:input_channels
                                                                    kernel_size:kernel_size
                                                                 constantValues:&constantValues
                                                                          width:output_width
                                                                         height:output_height
                                                                       stride_x:stride
                                                                       stride_y:stride];

  const size_t out_buffer_size = kernels_per_convolution * output_width * output_height;

  convolution.dataBuffer = [metalContext.device
      newBufferWithLength:sizeof(data_buffer_type) * input_height * input_width * input_channels
                  options:MTLStorageModeShared];

  convolution.outputBuffer = [metalContext.device
      newBufferWithLength:sizeof(data_buffer_type) * output_width * output_height * output_channels
                  options:MTLStorageModeShared];

  convolution.weightBuffer =
      [metalContext.device newBufferWithLength:sizeof(weight_buffer_type) * kernel_size *
                                               kernel_size * input_channels * output_channels
                                       options:MTLStorageModeShared];

  convolution.biasBuffer =
      [metalContext.device newBufferWithLength:sizeof(weight_buffer_type) * output_channels
                                       options:MTLStorageModeShared];

  dispatch_semaphore_t gpu_execution_done = dispatch_semaphore_create(0);

  for (int i = 0; i < iterations; i++) {
    for (int c = 0; c < convolutions; c++) {
      bool last_iteration = (c == convolutions - 1) && (i == iterations - 1);

      [convolution applyFilter:last_iteration ? ^(NSError *){
        dispatch_semaphore_signal(gpu_execution_done);
      } : (void(^)(NSError *)) NULL
            weightBufferOffset:sizeof(weight_buffer_type) * c * kernels_per_convolution * kernel_size * kernel_size * input_channels
            outputBufferOffset:sizeof(data_buffer_type) * c * kernels_per_convolution * output_width * output_height
       ];
    }
  }

  dispatch_semaphore_wait(gpu_execution_done, DISPATCH_TIME_FOREVER);
}

void BenchMetalConvolution(
    int inputC, int outputC, int kW, int kH, int stride, int inW, int inH, bool transposed) {
  testMetalConvolution(inW, inH, kW, inputC, outputC, 10);

  caffe2::Timer t;
  auto niter = 50;

  testMetalConvolution(inW, inH, kW, inputC, outputC, 50);

  printf("%s(%d -> %d, %dx%d - %dx%d - %s) took: %.4f ms/iter\n",
         "Conv",
         inputC,
         outputC,
         inW,
         inH,
         kW,
         kH,
         "METAL",
         t.MilliSeconds() / (float)niter);
}
