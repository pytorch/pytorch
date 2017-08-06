// Copyright 2004-present Facebook. All Rights Reserved.

#import "FBMetalConstantValues.h"
#import "FBMetalCNNConvolution.h"
#import "data_conversion.h"
#import "MetalShaderUtilities.h"

#include "caffe2/core/logging.h"

static constexpr size_t kThreadGroupSize_x = 4;
static constexpr size_t kThreadGroupSize_y = 8;

std::string FBMetalCNNConstantValues::to_string() {
  std::ostringstream ss;

  ss << "X:" <<
  input_channels << "x" <<
  input_width << "x" <<
  input_height << "[" <<
  input_stride_x << ":" <<
  input_stride_y << ":" <<
  input_pad_t << ":" <<
  input_pad_l << ":" <<
  input_pad_b << ":" <<
  input_pad_r << "]-Y:" <<
  output_channels << "x" <<
  output_width << "x" <<
  output_height << "-W:" <<
  filter_width << "x" <<
  filter_height << ":" <<
  (transposed ? "T" : "D");

  return ss.str();
}

namespace {
  extern const char *metalCode;
}

@interface FBMetalCNNConvolution () {
  int stride_x;
  int stride_y;
  uint input_batch_size;
  uint output_channels;
}
@end

@implementation FBMetalCNNConvolution

+ (id<MTLBuffer>)loadFilterWithImage:(const float*)weight_data
                        weightBuffer:(id<MTLBuffer>)weightBuffer
                             kernels:(NSUInteger)kernels
                       input_kernels:(NSUInteger)input_kernels
                       kernel_offset:(NSUInteger)kernel_offset
                       kernel_stride:(NSUInteger)kernel_stride
                            channels:(NSUInteger)channels
                               width:(NSUInteger)width
                              height:(NSUInteger)height
                          transposed:(bool)transposed
                             context:(MetalContext*)context

{
  reformatKernelImage<weight_buffer_type>(
      weight_data,
      kernels,
      input_kernels,
      kernel_offset,
      kernel_stride,
      channels,
      width,
      height,
      transposed,
      [&](size_t buffer_size) -> weight_buffer_type* {
        if (weightBuffer == nil || [weightBuffer length] != sizeof(weight_buffer_type) * buffer_size) {
          weightBuffer = [context.device newBufferWithLength:sizeof(weight_buffer_type) * buffer_size
                                                     options:MTLResourceOptionCPUCacheModeDefault];
          if (weightBuffer == nil) {
            LOG(ERROR) << "couldn't create weight buffer of size: " << buffer_size;
          }
        }

        return (weight_buffer_type*)(weightBuffer ? [weightBuffer contents] : NULL);
      });

  return weightBuffer;
}

- (void)loadFilterWithImage:(const float*)weight_data
                    kernels:(NSUInteger)kernels
              input_kernels:(NSUInteger)input_kernels
              kernel_offset:(NSUInteger)kernel_offset
              kernel_stride:(NSUInteger)kernel_stride
                   channels:(NSUInteger)channels
                      width:(NSUInteger)width
                     height:(NSUInteger)height
                 transposed:(bool)transposed {
  _weightBuffer = [FBMetalCNNConvolution loadFilterWithImage:weight_data
                                                weightBuffer:_weightBuffer
                                                     kernels:kernels
                                               input_kernels:input_kernels
                                               kernel_offset:kernel_offset
                                               kernel_stride:kernel_stride
                                                    channels:channels
                                                       width:width
                                                      height:height
                                                  transposed:transposed
                                                     context:self.context];
}

- (void)loadBiasData:(const float *)bias
              length:(NSUInteger)length {
  if (_biasBuffer == nil || [_biasBuffer length] != sizeof(weight_buffer_type) * length) {
    _biasBuffer = [self.context.device newBufferWithLength:sizeof(weight_buffer_type) * length
                                                   options:MTLResourceOptionCPUCacheModeDefault];
  }
  if (_biasBuffer) {
    weight_buffer_type *bias_data = (weight_buffer_type *) [_biasBuffer contents];
    for (int i = 0; i < length; i++) {
      bias_data[i] = bias[i];
    }
  }
}

+ (id<MTLBuffer>)loadDataWithImage:(const float*)imageData
                   imageDataBuffer:(id<MTLBuffer>)imageDataBuffer
                          channels:(NSUInteger)channels
                             width:(NSUInteger)width
                            height:(NSUInteger)height
                           context:(MetalContext*)context {
  id<MTLBuffer> newDataBuffer = imageDataBuffer;

  reformatInputImage<data_buffer_type>(
      imageData, channels, width, height, [&](size_t buffer_size) -> data_buffer_type* {
        if (newDataBuffer == nil || [newDataBuffer length] != sizeof(data_buffer_type) * buffer_size) {
          newDataBuffer = [context.device newBufferWithLength:sizeof(data_buffer_type) * buffer_size
                                                      options:MTLResourceOptionCPUCacheModeDefault];
          if (newDataBuffer == nil) {
            VLOG(0) << "couldn't create data buffer of size: " << buffer_size;
          }
        }

        return newDataBuffer ? (data_buffer_type*)[newDataBuffer contents] : NULL;
      });

  return newDataBuffer;
}

+ (instancetype)filterWithContext:(MetalContext*)context
                         channels:(NSUInteger)channels
                      kernel_size:(NSUInteger)kernel_size
                   constantValues:(FBMetalCNNConstantValues*)constantValues
                            width:(NSUInteger)width
                           height:(NSUInteger)height
                         stride_x:(NSUInteger)stride_x
                         stride_y:(NSUInteger)stride_y {
  return [[self alloc] initWithContext:context
                              channels:(NSUInteger)channels
                           kernel_size:kernel_size
                        constantValues:constantValues
                                 width:width
                                height:height
                              stride_x:stride_x
                              stride_y:stride_y];
}

- (MTLSize)threadsPerThreadgroup {
  if (input_batch_size > 1)
    return MTLSizeMake(input_batch_size * kThreadGroupSize_x, kThreadGroupSize_y / input_batch_size, 1);
  else
    return MTLSizeMake(kThreadGroupSize_x, kThreadGroupSize_y, 1);
}

- (MTLSize)threadgroupsPerGrid {
  MTLSize threadsPerThreadgroup = [self threadsPerThreadgroup];

  return MTLSizeMake(
                     ((input_batch_size * self.outputTextureDescriptor.width + stride_x - 1) / stride_x + threadsPerThreadgroup.width - 1) /
                     threadsPerThreadgroup.width,
                     ((self.outputTextureDescriptor.height + stride_y - 1) / stride_y + threadsPerThreadgroup.height - 1) /
                     threadsPerThreadgroup.height,
                     1);
}

// TODO: this code is temporary, we need to find the optimal strategy for large channel numbers

static uint input_channel_batching(uint input_channels) {
  return input_channels <= 32 ? 1 :
         input_channels % 8 == 0 ? 8 :
         input_channels % 4 == 0 ? 4 :
         input_channels % 3 == 0 ? 3 :
         input_channels % 2 == 0 ? 2 : 1;
}

- (instancetype)initWithContext:(MetalContext*)context
                       channels:(NSUInteger)channels
                    kernel_size:(NSUInteger)kernel_size
                 constantValues:(FBMetalCNNConstantValues*)constantValues
                          width:(NSUInteger)width
                         height:(NSUInteger)height
                       stride_x:(NSUInteger)_stride_x
                       stride_y:(NSUInteger)_stride_y {
  if ((self = [super initWithFunctionName:@"cnn_convolution_kern"
                              libraryName:@"Convolution"
                            librarySource:[NSString stringWithCString:metalCode encoding:NSUTF8StringEncoding]
                                  context:context
                           constantValues:constantValues])) {
    stride_x = _stride_x;
    stride_y = _stride_y;

    input_batch_size = input_channel_batching(constantValues->input_channels);
    output_channels = constantValues->output_channels;

    super.outputTextureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                       width:width
                                                                                      height:height
                                                                                   mipmapped:NO];
  }
  return self;
}

- (NSString*)replaceConstantValues:(FBMetalConstantValues *)constantValues
                     librarySource:(NSString*)librarySource {
  FBMetalCNNConstantValues* convolutionConstantValues = (FBMetalCNNConstantValues *) constantValues;
  std::string source = [librarySource UTF8String];

  REPLACE_CONSTANT(source, convolutionConstantValues->output_width,     0);
  REPLACE_CONSTANT(source, convolutionConstantValues->output_height,    1);
  REPLACE_CONSTANT(source, convolutionConstantValues->input_width,      2);
  REPLACE_CONSTANT(source, convolutionConstantValues->input_height,     3);
  REPLACE_CONSTANT(source, convolutionConstantValues->input_stride_x,   4);
  REPLACE_CONSTANT(source, convolutionConstantValues->input_stride_y,   5);
  REPLACE_CONSTANT(source, convolutionConstantValues->input_pad_t,      6);
  REPLACE_CONSTANT(source, convolutionConstantValues->input_pad_l,      7);
  REPLACE_CONSTANT(source, convolutionConstantValues->filter_width,     8);
  REPLACE_CONSTANT(source, convolutionConstantValues->filter_height,    9);
  REPLACE_CONSTANT(source, convolutionConstantValues->input_channels,   10);
  REPLACE_CONSTANT(source, convolutionConstantValues->output_channels,  11);
  REPLACE_CONSTANT(source, convolutionConstantValues->transposed,       12);
  REPLACE_CONSTANT(source, input_channel_batching(convolutionConstantValues->input_channels),  13);

  return [NSString stringWithUTF8String:source.c_str()];
}

// Bind data between C and Metal

- (void) configureArgumentTableWithCommandEncoder:(id<MTLComputeCommandEncoder>)commandEncoder
                               weightBufferOffset:(NSInteger)weightBufferOffset
                               outputBufferOffset:(NSInteger)outputBufferOffset {
  [commandEncoder setBuffer:_biasBuffer offset:0 atIndex:0];
  [commandEncoder setBuffer:_weightBuffer offset:weightBufferOffset atIndex:1];
  [commandEncoder setBuffer:_dataBuffer offset:0 atIndex:2];
  [commandEncoder setBuffer:_outputBuffer offset:outputBufferOffset atIndex:3];
}
@end

namespace {
const char *metalCode = R"Metal(
//  Copyright 2004-present Facebook. All Rights Reserved.

#include <metal_stdlib>

using namespace metal;

// These function_constant expressions are replaced at compile time with actual values

constant constexpr ushort output_size_x     [[ function_constant(0) ]];
constant constexpr ushort output_size_y     [[ function_constant(1) ]];
constant constexpr ushort input_size_x      [[ function_constant(2) ]];
constant constexpr ushort input_size_y      [[ function_constant(3) ]];
constant constexpr ushort input_stride_x    [[ function_constant(4) ]];
constant constexpr ushort input_stride_y    [[ function_constant(5) ]];
constant constexpr ushort input_pad_t       [[ function_constant(6) ]];
constant constexpr ushort input_pad_l       [[ function_constant(7) ]];
constant constexpr ushort filter_width      [[ function_constant(8) ]];
constant constexpr ushort filter_height     [[ function_constant(9) ]];
constant constexpr ushort input_channels    [[ function_constant(10) ]];
constant constexpr ushort output_channels   [[ function_constant(11) ]];
constant constexpr bool transposed          [[ function_constant(12) ]];
constant constexpr ushort input_batch_size  [[ function_constant(13) ]];

constant constexpr ushort2 input_padding = {
  transposed ? (filter_width - 1 - input_pad_l) : input_pad_l,
  transposed ? (filter_height - 1 - input_pad_t) : input_pad_t,
};

constant constexpr ushort DataSize      = output_channels <= 2 ? output_channels : 4;
constant constexpr ushort DataLength    = (output_channels + DataSize - 1) / DataSize;

typedef float data_buffer_type;

typedef vec<half, DataSize> vec_t;

typedef struct {
  data_buffer_type data[input_channels][input_size_y][input_size_x];
} input_data;

typedef struct {
  data_buffer_type data[output_channels][output_size_y][output_size_x];
} output_data;

typedef struct {
  vec_t data[input_channels][filter_height][filter_width][DataLength];
} filter_data;

typedef struct {
  half data[output_channels];
} bias_data;

typedef vec_t (thread_storage)[input_stride_y][input_stride_x][DataLength];

// Filter data sampler - returns an aligned array of vectors for the filter interleaved output channel data

template <typename T>
class planar_filter_sampler {
  const constant T (&channel_data)[filter_height][filter_width][DataLength];

 public:
  planar_filter_sampler(constant T data[input_channels][filter_height][filter_width][DataLength], ushort channel)
      : channel_data(data[channel]) {}

  typedef constant T (&return_type)[DataLength];

  return_type operator()(ushort2 off) {
    return channel_data[off.y][off.x];
  }
};

// Input data sampler - returns a scalar sample

template <typename T>
class planar_data_sampler {
  const constant T (&channel_data)[input_size_y][input_size_x];
  const ushort2     base;

 public:
  planar_data_sampler(const constant T data[input_channels][input_size_y][input_size_x], ushort2 _base, ushort channel)
      : channel_data(data[channel]), base(_base) {}

  inline T operator()(ushort2 off) {
    /*
     *  Note: Even though the documentation states otherwise, Metal will execute both sides of ternary statements,
     *        and, if the array index is out of bounds, it will crash.
     */

    const ushort2 idx = base + off;
    bool in_bounds = all(idx >= input_padding) &&
                     all(idx < (ushort2(input_size_x, input_size_y) * ushort2(input_stride_x, input_stride_y) + input_padding));
    const ushort2 padded_idx = (idx - (in_bounds ? input_padding : 0)) / ushort2(input_stride_x, input_stride_y);
    return in_bounds ? channel_data[padded_idx.y][padded_idx.x] : T(0);
  }
};

/*
 * To speed up data access in the convolution inner loop we use vector data access for the filter.
 * Every input pixel is multiplied by all the kernel values to generate all the output channels at once.
 */

template <ushort DataRemainder, typename T>
void accumulate(const ushort DataLength, T dst[], const constant T w[], const half sample) {
  for (ushort i = 0; i < DataLength; i++) {
    dst[i] += sample * w[i];
  }
}

template <>
void accumulate<3, half4>(const ushort DataLength, half4 dst[], const constant half4 w[], const half sample) {
  accumulate<0, half4>(DataLength - 1, dst, w, sample);
  dst[DataLength - 1].xyz += sample * w[DataLength - 1].xyz;
}

template <>
void accumulate<2, half4>(const ushort DataLength, half4 dst[], const constant half4 w[], const half sample) {
  accumulate<0, half4>(DataLength - 1, dst, w, sample);
  dst[DataLength - 1].xy += sample * w[DataLength - 1].xy;
}

template <>
void accumulate<1, half4>(const ushort DataLength, half4 dst[], const constant half4 w[], const half sample) {
  accumulate<0, half4>(DataLength - 1, dst, w, sample);
  dst[DataLength - 1].x += sample * w[DataLength - 1].x;
}

/*
 * Convolution inner loop: works for direct convolution (stride == 1) and for the transposed (stride > 1)
 */

template <typename T, typename S1, typename S2>
void convolution(T acc[input_stride_y][input_stride_x][DataLength], S1 in, S2 weights) {
  for (ushort y = 0; y < filter_height; y++) {
    for (ushort x = 0; x < filter_width; x++) {
      const ushort2 off(x, y);
      const ushort2 p0     = (input_padding + ushort2(input_stride_x, input_stride_y) - off) % ushort2(input_stride_x, input_stride_y);
      const half    sample = in(off + p0);
      auto          w      = weights(off);

      accumulate<output_channels % DataSize, T>(DataLength, acc[p0.y][p0.x], w, sample);
    }
  }
}

template <typename T>
void array_zero(T data[], ushort size) {
  for (ushort i = 0; i < size; i++) {
    data[i] = T(0);
  }
}

template <typename T1, typename T2>
void array_copy(T1 dst[], const T2 src[], ushort size) {
  for (ushort i = 0; i < size; i++) {
    dst[i] = src[i];
  }
}

template <typename T1, typename T2>
void array_add(T1 dst[], const T2 src[], ushort size) {
  for (ushort i = 0; i < size; i++) {
    dst[i] += src[i];
  }
}

template <ushort DataSize, typename T = vec<half, DataSize>>
const half element(const T data[], ushort i) {
  return data[i / DataSize][i % DataSize];
}

template <ushort DataSize, typename T = vec<half, DataSize>>
const half element(const threadgroup T data[], ushort i) {
  return data[i / DataSize][i % DataSize];
}

template <>
const half element<1, half>(const half data[], ushort i) {
  return data[i];
}

static void cnn_convolution_full(constant bias_data &bias,
                                 constant filter_data &filters,
                                 constant input_data &input,
                                 device output_data &output,
                                 ushort2 xgid)
{
  ushort2 gid = ushort2(input_stride_x, input_stride_y) * xgid;

  if (all(gid < ushort2(output_size_x, output_size_y))) {
    thread_storage storage;
    array_zero((thread vec_t *) storage, sizeof(thread_storage) / sizeof(vec_t));

    for (ushort channel = 0; channel < input_channels; channel++) {
      planar_data_sampler<data_buffer_type> in(input.data, gid, channel);
      planar_filter_sampler<vec_t> weights(filters.data, channel);

      convolution(storage, in, weights);
    }

    for (ushort c = 0; c < output_channels; c++) {
      for (ushort y = 0; y < input_stride_y; y++) {
        for (ushort x = 0; x < input_stride_x; x++) {
          ushort2 idx = gid + ushort2(x, y);
          if (all(idx < ushort2(output_size_x, output_size_y)))
            output.data[c][idx.y][idx.x] = element<DataSize, vec_t>(storage[y][x], c) + bias.data[c];
        }
      }
    }
  }
}

static void cnn_convolution_batch(constant bias_data &bias,
                                  constant filter_data &filters,
                                  constant input_data &input,
                                  device output_data &output,
                                  threadgroup thread_storage *batch_storage,
                                  ushort2 xgid)
{
  ushort2 gid = ushort2(input_stride_x, input_stride_y) * ushort2(xgid.x / input_batch_size, xgid.y);
  ushort batch = xgid.x % input_batch_size;

  if (gid.x < output_size_x && gid.y < output_size_y) {
    thread_storage storage;
    const ushort data_size = sizeof(thread_storage) / sizeof(vec_t);
    array_zero((thread vec_t *) storage, data_size);

    ushort base_channel = batch * input_channels / input_batch_size;
    for (ushort channel = 0; channel < input_channels / input_batch_size; channel++) {
      planar_data_sampler<data_buffer_type> in(input.data, gid, base_channel + channel);
      planar_filter_sampler<vec_t> weights(filters.data, base_channel + channel);

      convolution(storage, in, weights);
    }

    for (ushort b = 0; b < input_batch_size; b++) {
      threadgroup_barrier(mem_flags::mem_device);

      if (b == batch) {
        for (ushort y = 0; y < input_stride_y; y++) {
          for (ushort x = 0; x < input_stride_x; x++) {
            thread half *h_storage = (thread half *) storage[y][x];

            for (ushort c = 0; c < output_channels; c++) {
              ushort2 idx = gid + ushort2(x, y);
              if (all(idx < ushort2(output_size_x, output_size_y))) {
                device data_buffer_type *out_channel = &output.data[c][gid.y + y][gid.x + x];
                *out_channel = h_storage[c] + (b == 0 ? bias.data[c] : *out_channel);
              }
            }
          }
        }
      }
    }
  } else {
    for (ushort b = 0; b < input_batch_size; b++) {
      threadgroup_barrier(mem_flags::mem_device);
    }
  }
}

// clang-format off

kernel void cnn_convolution_kern(constant bias_data &bias             [[ buffer(0) ]],
                                 constant filter_data &filters        [[ buffer(1) ]],
                                 constant input_data &input           [[ buffer(2) ]],
                                 device output_data &output           [[ buffer(3) ]],
                                 threadgroup thread_storage *storage  [[ threadgroup(0) ]],
                                 ushort tid                           [[ thread_index_in_threadgroup ]],
                                 ushort2 xgid                         [[ thread_position_in_grid ]])

// clang-format on

{
  if (input_batch_size > 1) {
    ushort batch_index = (input_batch_size-1) * (tid / (input_batch_size));
    cnn_convolution_batch(bias, filters, input, output, &storage[batch_index], xgid);
  } else {
    cnn_convolution_full(bias, filters, input, output, xgid);
  }
}
)Metal";
}
