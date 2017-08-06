// Copyright 2004-present Facebook. All Rights Reserved.

#import "arm_neon_support.h"
#import "FBMetalCNNConvolution.h"
#import "metal_convolution.h"

#include "caffe2/core/logging.h"

@interface MetalConvolutionCacheEntry : NSObject
@property (nonatomic, strong) NSMutableArray<FBMetalCNNConvolution*>* convolutions;
@end
@implementation MetalConvolutionCacheEntry
@end

static MetalContext* metalContext = NULL;
static NSMutableDictionary<NSString*, MetalConvolutionCacheEntry*>* convolutionCache = NULL;

static void init_metal_pipeline() {
  if (metalContext == NULL) {
    metalContext     = [MetalContext getContext];
    convolutionCache = [NSMutableDictionary<NSString*, MetalConvolutionCacheEntry*> dictionary];
  }
}

bool calculate_kernels_per_convolution(int& kernels_per_convolution) {
  while (kernels_per_convolution > MAX_KERNELS_PER_CONVOLUTION) {
    if (kernels_per_convolution % 3 == 0)
    kernels_per_convolution /= 3;
    else if (kernels_per_convolution % 2 == 0)
    kernels_per_convolution /= 2;
    else {
      LOG(ERROR) << "The number of output channels must be a multiple of 2 or 3\n";
      return false;
    }
  }
  return true;
}

bool metal_convolution(
    id<MTLBuffer> inputBuffer,
    int           input_channels,
    int           input_width,
    int           input_height,
    int           input_stride_x,
    int           input_stride_y,
    int           input_pad_t,
    int           input_pad_l,
    int           input_pad_b,
    int           input_pad_r,
    id<MTLBuffer> weightBuffer,
    int           output_channels,
    int           kernel_channels,
    int           kernel_width,
    int           kernel_height,
    id<MTLBuffer> outputBuffer,
    int           output_number,
    int           output_width,
    int           output_height,
    const float*  bias,
    int           bias_length,
    bool          transposed) {
  init_metal_pipeline();

  if (transposed) {
    int t           = output_channels;
    output_channels         = kernel_channels;
    kernel_channels = t;
  }

  int output_batch_size = output_channels;

  if (!calculate_kernels_per_convolution(output_batch_size)) {
    return false;
  }

  FBMetalCNNConstantValues constantValues = FBMetalCNNConstantValues(
                                                                     input_width,
                                                                     input_height,
                                                                     input_channels,
                                                                     input_stride_x,
                                                                     input_stride_y,
                                                                     input_pad_t,
                                                                     input_pad_l,
                                                                     input_pad_b,
                                                                     input_pad_r,
                                                                     kernel_width,
                                                                     kernel_height,
                                                                     output_width,
                                                                     output_height,
                                                                     output_batch_size,
                                                                     transposed);

  NSString* key = [NSString stringWithUTF8String:constantValues.to_string().c_str()];

  const int batches = output_channels / output_batch_size;

  MetalConvolutionCacheEntry* cc = convolutionCache[key];

  if (cc == NULL) {
//    printf("metal_convolution_mtlbuffer: %s\n", [key UTF8String]);
    convolutionCache[key] = cc = [[MetalConvolutionCacheEntry alloc] init];
    cc.convolutions            = [NSMutableArray<FBMetalCNNConvolution*> arrayWithCapacity:batches];


    for (int c = 0; c < batches; c++) {
      FBMetalCNNConvolution* convolution = [FBMetalCNNConvolution filterWithContext:metalContext
                                                                           channels:input_channels
                                                                        kernel_size:kernel_height
                                                                     constantValues:&constantValues
                                                                              width:output_width
                                                                             height:output_height
                                                                           stride_x:input_stride_x
                                                                           stride_y:input_stride_y];
      cc.convolutions[c] = convolution;
    }
  }

  for (int c = 0; c < batches; c++) {
    FBMetalCNNConvolution* convolution = cc.convolutions[c];

    [convolution loadBiasData:&bias[output_batch_size * c] length:(NSUInteger)output_batch_size];

    convolution.dataBuffer   = inputBuffer;
    convolution.outputBuffer = outputBuffer;
    convolution.weightBuffer = weightBuffer;

    const int aligned_kernel_stride = output_batch_size <= 2 ? output_batch_size : 4 * ((output_batch_size + 3) / 4);
    const int buffer_size = aligned_kernel_stride * (output_batch_size / output_batch_size) * kernel_channels *
                            kernel_width * kernel_height;

    int weightBuffer_offset = c * sizeof(weight_buffer_type) * buffer_size;
    int outputBuffer_offset = c * sizeof(data_buffer_type) * output_batch_size * output_width * output_height;

    [convolution applyFilter:(void (^)(NSError* error)) nullptr
          weightBufferOffset:weightBuffer_offset
          outputBufferOffset:outputBuffer_offset];
  }

  return true;
}
