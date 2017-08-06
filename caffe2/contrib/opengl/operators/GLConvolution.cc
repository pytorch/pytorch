// Copyright 2004-present Facebook. All Rights Reserved.

#include "../core/GLContext.h"
#include "../core/GLFilter.h"
#include "../core/GLImage.h"
#include "../core/ImageAllocator.h"

#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/timer.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/conv_transpose_unpool_op_base.h"
#include <iostream>
#include <vector>

class GLConvolution : public GLFilter {
 public:
  static constexpr int MaxInputBatchSize = 8;
  static constexpr int MaxOutputBatchSize = 4;

  struct point {
    int x;
    int y;
  };

  struct descriptor {
    int input_channels;
    int output_channels;
    point kernel_size;
    point input_padding;
    point input_stride;
    bool transposed;
  };

  const float* kernel;
  const float* bias;
  const float* prelu_scale; // for PRelu

  binding* inputData[MaxInputBatchSize];
  binding* previousData[MaxOutputBatchSize];
  binding* kernelSize;
  binding* inputTileSize;
  binding* outputTileSize;
  binding* outputSize;
  binding* accumulate;
  binding* fusePRelu;
  binding* kernel_block[MaxInputBatchSize];
  binding* bias_block;
  binding* prelu_scale_block; // for PRelu

  const descriptor geometry;
  const int prelu_scale_size; // for PRelu
  const int input_batch_size;
  const int output_batch_size;
  const int input_tiles;
  const int output_tiles;

  static const char* fragment_shader;

  const std::vector<binding*> input_bindings(int input_batch_size, int output_batch_size) {
    std::vector<binding*> bindings({BINDING(inputTileSize),
                                    BINDING(outputTileSize),
                                    BINDING(outputSize),
                                    BINDING(accumulate),
                                    BINDING(kernelSize),
                                    BINDING(fusePRelu)});

    for (int i = 0; i < input_batch_size; i++) {
      bindings.push_back(inputData[i] = new binding{"inputData[" + caffe2::to_string(i) + "]"});
    }

    for (int i = 0; i < output_batch_size; i++) {
      bindings.push_back(previousData[i] =
                             new binding{"previousData[" + caffe2::to_string(i) + "]"});
    }

    return bindings;
  }

  const std::vector<binding*> uniform_blocks_bindings(int input_batch_size,
                                                      int output_batch_size,
                                                      bool fuse_prelu) {
    std::vector<binding*> bindings({BINDING(bias_block)});
    if (fuse_prelu) {
      bindings.push_back(BINDING(prelu_scale_block));
    }

    for (int i = 0; i < input_batch_size; i++) {
      bindings.push_back(kernel_block[i] =
                             new binding{"Kernel_block[" + caffe2::to_string(i) + "]"});
    }

    return bindings;
  }

  GLConvolution(const descriptor& _geometry,
                const float* _kernel,
                const float* _bias,
                const float* _prelu_scale = nullptr,
                int _prelu_scale_size = 0,
                int _input_batch_size = 1,
                int _output_batch_size = 1,
                int _input_tiles = 1,
                int _output_tiles = 1)
      : GLFilter(
            "GLConvolution",
            vertex_shader,
            fragment_shader,
            input_bindings(_input_batch_size, _output_batch_size),
            uniform_blocks_bindings(_input_batch_size, _output_batch_size, _prelu_scale != nullptr),
            {/* no attributes */},
            {{"KERNEL_SIZE_X", caffe2::to_string(_geometry.kernel_size.x)},
             {"KERNEL_SIZE_Y", caffe2::to_string(_geometry.kernel_size.y)},
             {"INPUT_BATCH_SIZE", caffe2::to_string(_input_batch_size)},
             {"OUTPUT_BATCH_SIZE", caffe2::to_string(_output_batch_size)},
             {"INPUT_TILES", caffe2::to_string(_input_tiles)},
             {"OUTPUT_TILES", caffe2::to_string(_output_tiles)},
             {"INPUT_PADDING_X",
              caffe2::to_string(_geometry.transposed
                                    ? _geometry.kernel_size.x - 1 - _geometry.input_padding.x
                                    : _geometry.input_padding.x)},
             {"INPUT_PADDING_Y",
              caffe2::to_string(_geometry.transposed
                                    ? _geometry.kernel_size.y - 1 - _geometry.input_padding.y
                                    : _geometry.input_padding.y)},
             {"INPUT_STRIDE_X", caffe2::to_string(_geometry.input_stride.x)},
             {"INPUT_STRIDE_Y", caffe2::to_string(_geometry.input_stride.y)},
             {"TRANSPOSED_CONVOLUTION", caffe2::to_string(_geometry.transposed)},
             {"TEXTURE_BORDER_CLAMP",
              caffe2::to_string(
                  GLContext::getGLContext()->GL_EXT_texture_border_clamp_defined())}}),
        kernel(_kernel),
        bias(_bias),
        prelu_scale(_prelu_scale),
        geometry(_geometry),
        prelu_scale_size(_prelu_scale_size),
        input_batch_size(_input_batch_size),
        output_batch_size(_output_batch_size),
        input_tiles(_input_tiles),
        output_tiles(_output_tiles) {
    if (input_tiles > 1 || output_tiles > 1) {
      printf("input_tiles: %d, output_tiles: %d\n", input_tiles, output_tiles);
    }

    int binding_point = 0;
    attach_uniform_buffer(bias_block, binding_point++, nullptr);

    for (int ob = 0; ob < input_batch_size; ob++) {
      attach_uniform_buffer(kernel_block[ob], binding_point++, nullptr);
    }
    if (_prelu_scale != nullptr) {
      attach_uniform_buffer(prelu_scale_block, binding_point++, nullptr);
    }
  }

  template <typename T>
  void convolution(const GLImageVector<T>& input_images, const GLImageVector<T>& output_images);

  void pack_kernel_data(float16_t* data,
                        size_t size,
                        int input_channels,
                        int output_channels,
                        int is,
                        int os,
                        int ib) {
    size_t kernel_batch_size = 4 * 4 * input_tiles * output_tiles * output_batch_size *
                               geometry.kernel_size.y * geometry.kernel_size.x * sizeof(float16_t);

    if (size != kernel_batch_size) {
      std::cerr << "size: " << size << ", kernel_batch_Size: " << kernel_batch_size << "\n";
      throw std::runtime_error("Kernel size mismatch");
    }

    typedef float16_t(packedKernel)[input_tiles * output_tiles * output_batch_size]
                                   [geometry.kernel_size.y][geometry.kernel_size.x][4][4];
    packedKernel& packed_kernel_data = *reinterpret_cast<packedKernel*>(data);

    const int batch_input_channels = std::min(4, input_channels - 4 * (is + ib));
    for (int ob = 0; ob < output_batch_size; ob++) {
      for (int it = 0; it < input_tiles; it++) {
        for (int ot = 0; ot < output_tiles; ot++) {
          const int batch_output_channels = std::min(4, output_channels - 4 * (os + ob));
          // printf("packing kernel data for slice %d, batch %d, output tile %d, input tile %d at
          // %p\n", os, ob, ot, it, &packed_kernel_data[input_tiles * output_tiles * ob +
          // output_tiles * it + ot][0][0][0][0]);
          for (int out = 0; out < batch_output_channels; out++) {
            for (int in = 0; in < batch_input_channels; in++) {
              for (int y = 0; y < geometry.kernel_size.y; y++) {
                for (int x = 0; x < geometry.kernel_size.x; x++) {
                  if (geometry.transposed) {
                    typedef float(kernelTensor)[input_channels][output_channels]
                                               [geometry.kernel_size.y][geometry.kernel_size.x];
                    const kernelTensor& kernel_data =
                        *reinterpret_cast<const kernelTensor*>(kernel);
                    packed_kernel_data[input_tiles * output_tiles * ob + output_tiles * it +
                                       ot][y][x][in][out] =
                        kernel_data[4 * (input_tiles * (is + ib) + it) + in]
                                   [4 * (output_tiles * (os + ob) + ot) + out]
                                   [geometry.kernel_size.y - 1 - y][geometry.kernel_size.x - 1 - x];
                  } else {
                    typedef float(kernelTensor)[output_channels][input_channels]
                                               [geometry.kernel_size.y][geometry.kernel_size.x];
                    const kernelTensor& kernel_data =
                        *reinterpret_cast<const kernelTensor*>(kernel);
                    float16_t kd = packed_kernel_data[input_tiles * output_tiles * ob +
                                                      output_tiles * it + ot][y][x][in][out] =
                        kernel_data[4 * (output_tiles * (os + ob) + ot) + out]
                                   [4 * (input_tiles * (is + ib) + it) + in][y][x];
                    // printf("%f, ", kd);
                  }
                }
              }
              // printf("\n");
            }
            // printf("\n");
          }
          // printf("\n");
        }
      }
    }
  }
};

// MARK: GLSL

const char* GLConvolution::fragment_shader = R"GLSL(#version 300 es

#define TRANSPOSED_CONVOLUTION      $(TRANSPOSED_CONVOLUTION)
#define INPUT_BATCH_SIZE            $(INPUT_BATCH_SIZE)
#define OUTPUT_BATCH_SIZE           $(OUTPUT_BATCH_SIZE)
#define INPUT_TILES                 $(INPUT_TILES)
#define OUTPUT_TILES                $(OUTPUT_TILES)
#define TEXTURE_BORDER_CLAMP        $(TEXTURE_BORDER_CLAMP)

precision mediump float;
precision mediump int;
precision mediump sampler2D;

in highp vec2 v_texCoord;

const ivec2 input_padding = ivec2($(INPUT_PADDING_X), $(INPUT_PADDING_Y));
const ivec2 input_stride = ivec2($(INPUT_STRIDE_X), $(INPUT_STRIDE_Y));
const ivec2 kernel_size = ivec2($(KERNEL_SIZE_X), $(KERNEL_SIZE_Y));
const int channels = 4;

uniform ivec2 kernelSize;
uniform ivec2 outputSize;
uniform ivec2 inputTileSize;
uniform ivec2 outputTileSize;
uniform bool accumulate;
uniform bool fusePRelu;

uniform sampler2D inputData[INPUT_BATCH_SIZE];
uniform sampler2D previousData[OUTPUT_BATCH_SIZE];

struct packedKernel {
  highp uvec4 packed_data[2];
};

#define unpackKernel(pk) \
  mat4(vec4(unpackHalf2x16(pk.packed_data[0].x), unpackHalf2x16(pk.packed_data[0].y)), \
       vec4(unpackHalf2x16(pk.packed_data[0].z), unpackHalf2x16(pk.packed_data[0].w)), \
       vec4(unpackHalf2x16(pk.packed_data[1].x), unpackHalf2x16(pk.packed_data[1].y)), \
       vec4(unpackHalf2x16(pk.packed_data[1].z), unpackHalf2x16(pk.packed_data[1].w)))

struct kernel {
  packedKernel data[kernel_size.x * kernel_size.y];
};

layout (std140) uniform Kernel_block {
  kernel kernel_data[INPUT_TILES * OUTPUT_TILES * OUTPUT_BATCH_SIZE];
} kernel_block[INPUT_BATCH_SIZE];

layout (std140) uniform bias_block {
  highp uvec4 bias[OUTPUT_TILES * (OUTPUT_BATCH_SIZE+1)/2];
};

layout (std140) uniform prelu_scale_block {
  highp uvec4 scale[OUTPUT_TILES * (OUTPUT_BATCH_SIZE+1)/2];
};

#define unpackHalf4x16(pd) vec4(unpackHalf2x16(pd.x), unpackHalf2x16(pd.y))

layout(location = 0) out mediump vec4 outputData0;
#if OUTPUT_BATCH_SIZE > 1
layout(location = 1) out mediump vec4 outputData1;
#if OUTPUT_BATCH_SIZE > 2
layout(location = 2) out mediump vec4 outputData2;
#if OUTPUT_BATCH_SIZE > 3
layout(location = 3) out mediump vec4 outputData3;
#endif
#endif
#endif

const bool no_bounds = bool(TEXTURE_BORDER_CLAMP) || all(equal(input_padding, ivec2(0)));

#define IN_BOUNDS(p, p0, p1) (all(greaterThanEqual(p, p0)) && all(lessThan(p, p1)))

#if TRANSPOSED_CONVOLUTION

#define CONVOLUTION(ib) { \
  ivec2 p0 = (input_padding + input_stride - tileCoord % input_stride) % input_stride; \
  for (int y = p0.y; y < kernel_size.y; y+=input_stride.y) { \
    for (int x = p0.x; x < kernel_size.x; x+=input_stride.x) { \
      ivec2 idx = tileCoord + ivec2(x, y) - input_padding; \
      if (no_bounds || IN_BOUNDS(idx, ivec2(0), inputTileSize * input_stride)) { \
        vec4 data = texelFetch(inputData[ib], inputTileOffset + idx / input_stride, 0); \
        for (int ob = 0; ob < OUTPUT_BATCH_SIZE; ob++) { \
          mediump mat4 k = unpackKernel(kernel_block[ib].kernel_data[INPUT_TILES * OUTPUT_TILES * ob + kernelIdx].data[y * kernelSize.x + x]); \
          sum[ob] += k * data; \
        } \
      } \
    } \
  } \
}

#else

#define CONVOLUTION(ib) { \
  for (int y = 0, i = 0; y < kernelSize.y; y++) { \
    for (int x = 0; x < kernelSize.x; x++, i++) { \
      ivec2 idx = tileCoord + ivec2(x, y); \
      if (no_bounds || IN_BOUNDS(idx, ivec2(0), inputTileSize)) { \
        vec4 data = texelFetch(inputData[ib], inputTileOffset + idx, 0); \
        for (int ob = 0; ob < OUTPUT_BATCH_SIZE; ob++) { \
          mediump mat4 k = unpackKernel(kernel_block[ib].kernel_data[INPUT_TILES * OUTPUT_TILES * ob + kernelIdx].data[i]); \
          sum[ob] += k * data; \
        } \
      } \
    } \
  } \
}

#endif

#define TILED_CONVOLUTION ((INPUT_TILES > 1) || (OUTPUT_TILES > 1))

void main() {
  ivec2 inputSize = textureSize(inputData[0], 0);
  ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize));

#if TILED_CONVOLUTION
  ivec2 inputTiles = inputSize / inputTileSize;
  ivec2 tile = texelCoord / outputTileSize;
  ivec2 tileCoord = texelCoord % outputTileSize;
  int tileNum = 2 * tile.y + tile.x;
#define LOOP_PRECISION highp
#else
  const ivec2 inputTiles = ivec2(1, 1);
  const ivec2 tile = ivec2(0, 0);
  ivec2 tileCoord = texelCoord;
  const int tileNum = 0;
#define LOOP_PRECISION mediump
#endif

#if !TRANSPOSED_CONVOLUTION
  tileCoord = input_stride * tileCoord - input_padding;
#endif

  LOOP_PRECISION vec4 sum[OUTPUT_BATCH_SIZE] = vec4[OUTPUT_BATCH_SIZE](vec4(0)
#if OUTPUT_BATCH_SIZE > 1
                                                                       , vec4(0)
#if OUTPUT_BATCH_SIZE > 2
                                                                       , vec4(0)
#if OUTPUT_BATCH_SIZE > 3
                                                                       , vec4(0)
#endif
#endif
#endif
                                                                       );
#if TILED_CONVOLUTION
  for (int inTileY = 0; inTileY < inputTiles.y; inTileY++) {
    for (int inTileX = 0; inTileX < inputTiles.x; inTileX++) {
      int inTileId = inTileY * inputTiles.x + inTileX;
      int kernelIdx = OUTPUT_TILES * inTileId + tileNum;
      ivec2 inputTileOffset = ivec2(inTileX, inTileY) * inputTileSize;
#else
  const int kernelIdx = 0;
  const ivec2 inputTileOffset = ivec2(0, 0);
#endif

      CONVOLUTION(0);
#if INPUT_BATCH_SIZE > 1
      CONVOLUTION(1);
#if INPUT_BATCH_SIZE > 2
      CONVOLUTION(2);
#if INPUT_BATCH_SIZE > 3
      CONVOLUTION(3);
#if INPUT_BATCH_SIZE > 4
      CONVOLUTION(4);
#if INPUT_BATCH_SIZE > 5
      CONVOLUTION(5);
#if INPUT_BATCH_SIZE > 6
      CONVOLUTION(6);
#if INPUT_BATCH_SIZE > 7
      CONVOLUTION(7);
#endif
#endif
#endif
#endif
#endif
#endif
#endif

#if TILED_CONVOLUTION
    }
  }
#endif

  vec4 biasValue = (tileNum % 2 == 0) ? unpackHalf4x16(bias[tileNum/2].xy) : unpackHalf4x16(bias[tileNum/2].zw);

  vec4 value = sum[0] + (accumulate ? texelFetch(previousData[0], texelCoord, 0) : biasValue);
  outputData0 = fusePRelu ? mix(value * unpackHalf4x16(scale[0].xy), value, vec4(greaterThan(value, vec4(0)))) : value;
#if OUTPUT_BATCH_SIZE > 1
  value = sum[1] + (accumulate ? texelFetch(previousData[1], texelCoord, 0) : unpackHalf4x16(bias[0].zw));
  outputData1 = fusePRelu ? mix(value * unpackHalf4x16(scale[0].zw), value, vec4(greaterThan(value, vec4(0)))) : value;
#if OUTPUT_BATCH_SIZE > 2
  value = sum[2] + (accumulate ? texelFetch(previousData[2], texelCoord, 0) : unpackHalf4x16(bias[1].xy));
  outputData2 = fusePRelu ? mix(value * unpackHalf4x16(scale[1].xy), value, vec4(greaterThan(value, vec4(0)))) : value;
#if OUTPUT_BATCH_SIZE > 3
  value = sum[3] + (accumulate ? texelFetch(previousData[3], texelCoord, 0) : unpackHalf4x16(bias[1].zw));
  outputData3 = fusePRelu ? mix(value * unpackHalf4x16(scale[1].zw), value, vec4(greaterThan(value, vec4(0)))) : value;
#endif
#endif
#endif
}

)GLSL";

template <typename T>
void GLConvolution::convolution(const GLImageVector<T>& input_images,
                                const GLImageVector<T>& output_images) {
  for (int i = 0; i < input_images.size(); i++) {
    GLImage<T>* input_image = input_images[i];
    GLImage<T>* output_image = output_images[i];
    int input_slices = input_image->slices;
    int output_slices = output_image->slices;
    int input_tiles = input_image->tile_x * input_image->tile_y;
    int output_tiles = output_image->tile_x * output_image->tile_y;

    for (int is = 0; is < input_slices; is += input_batch_size) {
      for (int os = 0; os < output_slices; os += output_batch_size) {
        gl_log(GL_VERBOSE, "GLConvolution::convolution - is: %d, os: %d\n", is, os);
        int binding_point = 0;
        attach_uniform_buffer(bias_block, binding_point++, [&](float16_t* data, size_t size) {
          if (size != output_tiles * (4 * ((output_batch_size + 1) / 2 * 2) * sizeof(float16_t))) {
            std::cerr << "size: " << size
                      << ", (4 * ((output_batch_size+1)/2 * 2) * sizeof(float16_t)): "
                      << output_tiles * (4 * ((output_batch_size + 1) / 2 * 2) * sizeof(float16_t))
                      << "\n";
            throw std::runtime_error("Bias size mismatch");
          }

          for (int ib = 0; ib < std::min(4 * output_tiles * output_batch_size,
                                         geometry.output_channels - 4 * os);
               ib++) {
            data[ib] = bias[4 * os + ib];
          }
        });

        for (int ib = 0; ib < input_batch_size; ib++) {
          attach_uniform_buffer(
              kernel_block[ib], binding_point++, [&](float16_t* data, size_t size) {
                pack_kernel_data(data,
                                 size,
                                 input_image->channels,
                                 output_image->channels,
                                 is,
                                 input_tiles * output_tiles * os,
                                 ib);
              });
        }

        // PRelu scale
        if (prelu_scale != nullptr && is == input_slices - input_batch_size) {
          attach_uniform_buffer(
              prelu_scale_block, binding_point++, [&](float16_t* data, size_t size) {
                for (int i = 0;
                     i < std::min(4 * output_batch_size, geometry.output_channels - 4 * os);
                     i++) {
                  data[i] = prelu_scale_size == geometry.output_channels ? prelu_scale[4 * os + i]
                                                                         : prelu_scale[0];
                }
              });
        }

        std::vector<texture_attachment> input_attachments;
        for (int ib = 0; ib < input_batch_size; ib++) {
          input_attachments.push_back({input_image->textures[is + ib], inputData[ib]});
        }
        for (int ib = 0; ib < output_batch_size; ib++) {
          input_attachments.push_back({output_image->textures[os + ib], previousData[ib]});
        }

        run(input_attachments,
            {output_image->textures.begin() + os,
             output_image->textures.begin() + os + output_batch_size},
            [&]() {
              glUniform2i(inputTileSize->location, input_image->width, input_image->height);
              glUniform2i(outputTileSize->location, output_image->width, output_image->height);
              glUniform2i(outputSize->location,
                          output_image->width * output_image->tile_x,
                          output_image->height * output_image->tile_y);
              glUniform1i(accumulate->location, is != 0);
              glUniform1i(fusePRelu->location,
                          prelu_scale != nullptr && is == input_slices - input_batch_size);
              glUniform2i(kernelSize->location, geometry.kernel_size.x, geometry.kernel_size.y);
            },
            output_image->width * output_image->tile_x,
            output_image->height * output_image->tile_y);
      }
    }
  }
}

namespace caffe2 {

template <typename OPBase>
static void computeOutputHW(OPBase* op, int H, int W, int* OH, int* OW) {
  Tensor<CPUContext> input, output;
  input.Resize(1, 1, H, W);
  op->SetOutputSize(input, &output, 1);
  CAFFE_ENFORCE_EQ(output.ndim(), 4);
  *OH = output.dim(2);
  *OW = output.dim(3);
}

// Todo: optimize input/output batch size and use of uniforms/textures for kernel data
static void computeBatchSizes(GLConvolution::descriptor& geometry,
                              int& input_batch_size,
                              int& output_batch_size) {
  int kernel_size = std::max(geometry.kernel_size.x, geometry.kernel_size.y);
  int input_slices = (geometry.input_channels + 3) / 4;
  int output_slices = (geometry.output_channels + 3) / 4;

#if CAFFE2_ANDROID
  input_batch_size = input_slices % 2 == 0 ? 2 : 1;
  output_batch_size = output_slices % 2 == 0 ? 2 : 1;
#else
  if (iPhoneVersion() >= 8) {
    // iPhone 6S and up
    input_batch_size =
        input_slices % 8 == 0
            ? 8
            : input_slices % 4 == 0 ? 4 : input_slices % 3 == 0 ? 3 : input_slices % 2 == 0 ? 2 : 1;
    output_batch_size =
        output_slices % 4 == 0 ? 4 : output_slices % 3 == 0 ? 3 : output_slices % 2 == 0 ? 2 : 1;
  }
#endif
}

template <class T, bool fusePRelu, bool fuseRelu>
class OpenGLConvOp final : public ConvPoolOpBase<CPUContext>, ImageAllocator<T> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  OpenGLConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW, "OpenGL only supports NCHW order.");
    OPERATOR_NEEDS_FEATURE(group_ == 1, "OpenGL only supports group == 1");
    OPERATOR_NEEDS_FEATURE(dilation_h() == 1 && dilation_w() == 1,
                           "OpenGL only supports dialation == 1");
  }

  bool RunOnDeviceWithOrderNCHW() override {
    const GLImageVector<T>& input = Inputs()[INPUT]->template Get<GLImageVector<T>>();
    auto& filter = Input(FILTER);
    auto& bias = Input(BIAS);

    const int num_images = input.size();
    const int input_channels = input.channels();
    const int input_width = input.width();
    const int input_height = input.height();

    CAFFE_ENFORCE(filter.ndim(), 4);
    const int M = filter.dim32(0);
    const int kernel_width = filter.dim32(2);
    const int kernel_height = filter.dim32(3);

    CAFFE_ENFORCE(filter.dim32(1) == input_channels, "");
    CAFFE_ENFORCE(filter.dim32(2) == kernel_h(), "");
    CAFFE_ENFORCE(filter.dim32(3) == kernel_w(), "");
    CAFFE_ENFORCE(bias.ndim() == 1, "");
    CAFFE_ENFORCE(bias.dim32(0) == M, "");

    int output_height;
    int output_width;
    const int output_channels = M;
    computeOutputHW(this, input_height, input_width, &output_height, &output_width);

    const float* prelu_scale = nullptr;
    int prelu_scale_size = 0;
    if (fusePRelu) {
      auto& prelu = Input(PRELU);
      prelu_scale = prelu.template data<float>();
      prelu_scale_size = prelu.size();
    } else if (fuseRelu) {
      float val = 0;
      prelu_scale = &val;
      prelu_scale_size = 1;
    }

    int is_last = GetSingleArgument<int>("is_last", 0);

    GLImageVector<T>* output = ImageAllocator<T>::newImage(num_images,
                                                           output_width,
                                                           output_height,
                                                           output_channels,
                                                           input.tile_x(),
                                                           input.tile_y(),
                                                           is_last);

    // TODO: figure out the dilation business
    GLConvolution::descriptor geometry{input_channels,
                                       output_channels,
                                       {kernel_width, kernel_height},
                                       {pad_l(), pad_t()},
                                       {stride_w(), stride_h()},
                                       false};

    if (!conv) {
      int input_batch_size = 1, output_batch_size = 1;
      if (input.tile_x() == 1 && input.tile_y() == 1) {
        computeBatchSizes(geometry, input_batch_size, output_batch_size);
      }

      input_batch_size = GetSingleArgument<int>("input_batch_size", input_batch_size);
      output_batch_size = GetSingleArgument<int>("output_batch_size", output_batch_size);

      LOG(INFO) << input_channels << ": " << input_height << " X " << input_width << " => "
                << output_channels << ": " << output_height << " X " << output_width
                << " Kernel: " << kernel_width << "X" << kernel_height;
      LOG(INFO) << "input_batch_size = " << input_batch_size
                << ", output_batch_size = " << output_batch_size;

      conv.reset(new GLConvolution(geometry,
                                   filter.template data<float>(),
                                   bias.template data<float>(),
                                   prelu_scale,
                                   prelu_scale_size,
                                   input_batch_size,
                                   output_batch_size,
                                   input.tile_x() * input.tile_y(),
                                   output->tile_x() * output->tile_y()));
    }

    conv->convolution(input, *output);

    Outputs()[0]->Reset(output);

    return true;
  }

 private:
  std::unique_ptr<GLConvolution> conv;

  INPUT_TAGS(INPUT, FILTER, BIAS, PRELU);
};

REGISTER_CPU_OPERATOR(OpenGLConv, OpenGLConvOp<float16_t, false, false>);
OPERATOR_SCHEMA(OpenGLConv).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(OpenGLConvPRelu, OpenGLConvOp<float16_t, true, false>);
OPERATOR_SCHEMA(OpenGLConvPRelu).NumInputs(4).NumOutputs(1);

REGISTER_CPU_OPERATOR(OpenGLConvRelu, OpenGLConvOp<float16_t, false, true>);
OPERATOR_SCHEMA(OpenGLConvRelu).NumInputs(3).NumOutputs(1);

template <class T, bool fusePRelu, bool fuseRelu>
class OpenGLConvTransposeOp final : public ConvTransposeUnpoolBase<CPUContext>, ImageAllocator<T> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  OpenGLConvTransposeOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvTransposeUnpoolBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW, "OpenGL only supports NCHW order.");
    OPERATOR_NEEDS_FEATURE(adj_h_ == 0 && adj_w_ == 0,
                           "OpenGL only supports adj_h == 1 and adj_w == 1");
  }

  bool RunOnDeviceWithOrderNCHW() override {
    const GLImageVector<T>& input = Inputs()[INPUT]->template Get<GLImageVector<T>>();
    auto& filter = Input(FILTER);
    auto& bias = Input(BIAS);

    const int num_images = input.size();
    const int input_channels = input.channels();
    const int input_width = input.width();
    const int input_height = input.height();

    CAFFE_ENFORCE(filter.ndim() == 4, "filter must be 4D tensor");
    const int M = filter.dim32(0);
    const int C = filter.dim32(1);
    const int kernel_width = filter.dim32(2);
    const int kernel_height = filter.dim32(3);

    CAFFE_ENFORCE(input_channels == M, "filter number must be equal to input channel number");
    CAFFE_ENFORCE(filter.dim32(2) == kernel_h_, "filter height must be equal to kernel height");
    CAFFE_ENFORCE(filter.dim32(3) == kernel_w_, "filter width must be equal to kernel width");
    CAFFE_ENFORCE(bias.ndim() == 1, "bias must be 1D tensor");
    CAFFE_ENFORCE(bias.dim32(0) == C, "bias dimension must be equal to output channel number");

    int output_height;
    int output_width;
    const int output_channels = C;
    computeOutputHW(this, input_height, input_width, &output_height, &output_width);

    const float* prelu_scale = nullptr;
    int prelu_scale_size = 0;
    if (fusePRelu) {
      auto& prelu = Input(PRELU);
      prelu_scale = prelu.template data<float>();
      prelu_scale_size = prelu.size();
    } else if (fuseRelu) {
      float val = 0;
      prelu_scale = &val;
      prelu_scale_size = 1;
    }

    int is_last = GetSingleArgument<int>("is_last", 0);

    GLImageVector<T>* output = ImageAllocator<T>::newImage(num_images,
                                                           output_width,
                                                           output_height,
                                                           output_channels,
                                                           input.tile_x(),
                                                           input.tile_y(),
                                                           is_last);

    // TODO: figure out the adj business
    GLConvolution::descriptor geometry{input_channels,
                                       output_channels,
                                       {kernel_width, kernel_height},
                                       {pad_l_, pad_t_},
                                       {stride_w_, stride_h_},
                                       true};

    if (!conv) {
      int input_batch_size = 1, output_batch_size = 1;
      if (input.tile_x() == 1 && input.tile_y() == 1) {
        computeBatchSizes(geometry, input_batch_size, output_batch_size);
      }

      input_batch_size = GetSingleArgument<int>("input_batch_size", input_batch_size);
      output_batch_size = GetSingleArgument<int>("output_batch_size", output_batch_size);

      LOG(INFO) << input_channels << ": " << input_height << " X " << input_width << " => "
                << output_channels << ": " << output_height << " X " << output_width
                << " Kernel: " << kernel_width << "X" << kernel_height;

      LOG(INFO) << "input_batch_size = " << input_batch_size
                << ", output_batch_size = " << output_batch_size;

      conv.reset(new GLConvolution(geometry,
                                   filter.template data<float>(),
                                   bias.template data<float>(),
                                   prelu_scale,
                                   prelu_scale_size,
                                   input_batch_size,
                                   output_batch_size,
                                   input.tile_x() * input.tile_y(),
                                   output->tile_x() * output->tile_y()));
    }

    conv->convolution(input, *output);

    Outputs()[0]->Reset(output);

    return true;
  }

 private:
  std::unique_ptr<GLConvolution> conv;

  INPUT_TAGS(INPUT, FILTER, BIAS, PRELU);
};

REGISTER_CPU_OPERATOR(OpenGLConvTranspose, OpenGLConvTransposeOp<float16_t, false, false>);
OPERATOR_SCHEMA(OpenGLConvTranspose).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(OpenGLConvTransposePRelu, OpenGLConvTransposeOp<float16_t, true, false>);
OPERATOR_SCHEMA(OpenGLConvTransposePRelu).NumInputs(4).NumOutputs(1);

REGISTER_CPU_OPERATOR(OpenGLConvTransposeRelu, OpenGLConvTransposeOp<float16_t, false, true>);
OPERATOR_SCHEMA(OpenGLConvTransposeRelu).NumInputs(3).NumOutputs(1);
} // namespace caffe2
