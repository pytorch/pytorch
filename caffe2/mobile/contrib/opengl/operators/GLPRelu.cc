
#include "../core/GLFilter.h"
#include "../core/GLImage.h"
#include "../core/ImageAllocator.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include <iostream>
#include <vector>

class GLPRelu : public GLFilter {
 public:
  typedef enum { PRelu = 0, Relu = 1 } ReluType;

  const float* scale;

  binding* inputData;
  binding* scale_block;

  const int scale_size;
  const int channels;
  const int output_tile_x;
  const int output_tile_y;
  const int output_tile_width;
  const int output_tile_height;

  GLPRelu(
      const float* _scale,
      const int _scale_size,
      const int _channels,
      int _output_tile_x,
      int _output_tile_y,
      int _output_tile_width,
      int _output_tile_height)
      : GLFilter(
            "GLPRelu",
            vertex_shader,
            fragment_shader,
            std::vector<binding*>({BINDING(inputData)}),
            std::vector<binding*>({BINDING(scale_block)}),
            {/* no attributes */},
            {{"USE_RELU", caffe2::to_string(PRelu)},
             {"OUTPUT_TILES",
              caffe2::to_string(_output_tile_x * _output_tile_y)},
             {"OUTPUT_TILE_X", caffe2::to_string(_output_tile_x)},
             {"OUTPUT_TILE_WIDTH", caffe2::to_string(_output_tile_width)},
             {"OUTPUT_TILE_HEIGHT", caffe2::to_string(_output_tile_height)},
             {"TILED_PRELU",
              caffe2::to_string(_output_tile_x > 1 || _output_tile_y > 1)}}),
        scale(_scale),
        scale_size(_scale_size),
        channels(_channels),
        output_tile_x(_output_tile_x),
        output_tile_y(_output_tile_y),
        output_tile_width(_output_tile_width),
        output_tile_height(_output_tile_height) {}

  GLPRelu(const int _channels)
      : GLFilter("GLRelu",
                 vertex_shader,
                 fragment_shader,
                 std::vector<binding*>({BINDING(inputData)}),
                 {/* no uniform blocks */},
                 {/* no attributes */},
                 {{"USE_RELU", caffe2::to_string(Relu)},
                  {"OUTPUT_TILES", caffe2::to_string(1)},
                  {"OUTPUT_TILE_X", caffe2::to_string(1)},
                  {"OUTPUT_TILE_WIDTH", caffe2::to_string(1)},
                  {"OUTPUT_TILE_HEIGHT", caffe2::to_string(1)},
                  {"TILED_PRELU", caffe2::to_string(0)}}),
        scale(nullptr),
        scale_block(nullptr),
        scale_size(0),
        channels(_channels),
        output_tile_x(1),
        output_tile_y(1),
        output_tile_width(1),
        output_tile_height(1) {}

  template <typename T>
  void prelu(const GLImageVector<T>& input_images,
             const GLImageVector<T>& output_images,
             GLPRelu::ReluType reluType);

  static const char* fragment_shader;
};

// MARK: GLSL

const char* GLPRelu::fragment_shader = R"GLSL(#version 300 es
#define TILED_PRELU                 $(TILED_PRELU)
#define USE_RELU                    $(USE_RELU)

// tiling
#define OUTPUT_TILES                $(OUTPUT_TILES)
#define OUTPUT_TILE_X               $(OUTPUT_TILE_X)
#define OUTPUT_TILE_WIDTH           $(OUTPUT_TILE_WIDTH)
#define OUTPUT_TILE_HEIGHT          $(OUTPUT_TILE_HEIGHT)

// common
precision mediump float;
precision highp int;

TEXTURE_INPUT(inputData);
TEXTURE_OUTPUT(0, outputData);

in highp vec2 v_texCoord;

#if USE_RELU

// Relu
void main() {
  ivec2 inputSize = textureSize(inputData, 0);
  ivec2 texelCoord = ivec2(v_texCoord * vec2(inputSize));
  vec4 value = TEXTURE_LOAD(inputData, texelCoord);
  outputData = TEXTURE_STORE(max(value, vec4(0.0)));
}

#else

#if TILED_PRELU
const ivec2 outputTileSize = ivec2(OUTPUT_TILE_WIDTH, OUTPUT_TILE_HEIGHT);

layout (std140) uniform scale_block {
  highp uvec4 scale[(OUTPUT_TILES + 1) / 2];
};

void main() {
  ivec2 inputSize = textureSize(inputData, 0);
  ivec2 texelCoord = ivec2(v_texCoord * vec2(inputSize));

  ivec2 tile = texelCoord / outputTileSize; // 2D output tile idx
  int tileNum = OUTPUT_TILE_X * tile.y + tile.x; // 1D output tile idx

  // outputData = value > 0 ? value : value * weight;
  vec4 value = TEXTURE_LOAD(inputData, texelCoord);
  vec4 preluValue = (tileNum % 2 == 0) ? unpackHalf4x16(scale[tileNum/2].xy) : unpackHalf4x16(scale[tileNum/2].zw);
  value = mix(value * preluValue, value, vec4(greaterThan(value, vec4(0))));
  outputData = TEXTURE_STORE(value);
}
#else
layout (std140) uniform scale_block {
  highp uvec4 scale;
};
void main() {
  ivec2 inputSize = textureSize(inputData, 0);
  ivec2 texelCoord = ivec2(v_texCoord * vec2(inputSize));

  // outputData = value > 0 ? value : value * weight;
  vec4 value = TEXTURE_LOAD(inputData, texelCoord);
  value = mix(value * unpackHalf4x16(scale.xy), value, vec4(greaterThan(value, vec4(0))));
  outputData = TEXTURE_STORE(value);
}
#endif // TILED_PRELU

#endif // USE_RELU

)GLSL";

template <typename T>
void GLPRelu::prelu(const GLImageVector<T>& input_images,
                    const GLImageVector<T>& output_images,
                    GLPRelu::ReluType reluType) {
  int num_images = input_images.size();
  for (int i = 0; i < num_images; i++) {
    GLImage<T>* input_image = input_images[i];
    GLImage<T>* output_image = output_images[i];
    int input_slices = input_image->slices;
    int output_slices = output_image->slices;

    for (int is = 0; is < input_slices; is++) {
      if (reluType == PRelu) {
        attach_uniform_buffer<float16_t>(scale_block, 0, [&](float16_t* data, size_t size) {
          int output_tiles = output_tile_x * output_tile_y;
          for (int j = 0, k = 4 * is * output_tiles;
               k < std::min(channels, 4 * (is + 1) * output_tiles);
               j++, k++) {
            data[j] = scale_size == channels ? scale[k] : scale[0];
          }
        });
      }

      std::vector<texture_attachment> input_attachments;

      input_attachments.push_back({input_image->textures[is], inputData});

      run(input_attachments,
          {output_image->textures.begin() + is, output_image->textures.begin() + is + 1},
          [&]() {},
          output_image->texture_width,
          output_image->texture_height);
    }
  }
}

namespace caffe2 {
template <typename T, GLPRelu::ReluType reluType>
class OpenGLPReluOp final : public Operator<CPUContext>, ImageAllocator<T> {
 public:
  OpenGLPReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        order_(StringToStorageOrder(OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW, "OpenGL only supports NCHW order.");
  }

  bool RunOnDevice() override {
    const GLImageVector<T>& input = Inputs()[0]->template Get<GLImageVector<T>>();
    const int num_images = input.size();
    const int input_channels = input.channels();
    const int input_width = input.width();
    const int input_height = input.height();

    const int output_channels = input_channels;
    const int output_width = input_width;
    const int output_height = input_height;

    int is_last = OperatorBase::GetSingleArgument<int>("is_last", 0);

    const int input_tile_x = input.tile_x(), input_tile_y = input.tile_y();
    const int output_tile_x = input_tile_x, output_tile_y = input_tile_y;
    if (input_tile_x > 1 || input_tile_y > 1) {
      CAFFE_ENFORCE_EQ(input.slices(), 1, "Input needs to be tiled in a single texture");
    }

    GLImageVector<T>* output = ImageAllocator<T>::newImage(num_images,
                                                           output_width,
                                                           output_height,
                                                           output_channels,
                                                           output_tile_x,
                                                           output_tile_y,
                                                           is_last);

    const auto* scale = reluType == GLPRelu::PRelu ? &Input(1) : nullptr;

    if (!_prelu) {
      if (reluType == GLPRelu::PRelu) {
        _prelu.reset(new GLPRelu(scale->template data<float>(),
                                 scale->size(),
                                 input_channels,
                                 output_tile_x,
                                 output_tile_y,
                                 output_width,
                                 output_height));
      } else {
        _prelu.reset(new GLPRelu(input_channels));
      }
    }

    _prelu->prelu(input, *output, reluType);

    Outputs()[0]->Reset(output);

    return true;
  }

 private:
  StorageOrder order_;
  std::unique_ptr<GLPRelu> _prelu;
};

REGISTER_CPU_OPERATOR(OpenGLPRelu, OpenGLPReluOp<float16_t, GLPRelu::PRelu>);
OPERATOR_SCHEMA(OpenGLPRelu)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape();
REGISTER_CPU_OPERATOR(OpenGLRelu, OpenGLPReluOp<float16_t, GLPRelu::Relu>);
OPERATOR_SCHEMA(OpenGLRelu)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape();
} // namespace caffe2
