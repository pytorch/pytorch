
#include "../core/GLFilter.h"
#include "../core/GLImage.h"
#include "../core/ImageAllocator.h"
#include "gl_tiling_utils.h"

#include <iostream>
#include <vector>
#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "caffe2/utils/math.h"

class GLConcat : public GLFilter {
 public:
  bool tiling_;
  binding* inputData;
  binding* outputSize;
  binding* inputTileRange;
  binding* input_tile_x;

  GLConcat(tile_descriptor output_tile_geometries, bool tiling = false)
      : GLFilter("GLConcat",
                 vertex_shader,
                 fragment_shader,
                 std::vector<binding*>(
                     {BINDING(outputSize), BINDING(inputData), BINDING(inputTileRange), BINDING(input_tile_x)}),
                 {/* no uniform blocks */},
                 {/* no attributes */},
                 {{"TILING", caffe2::to_string(tiling)},
                  {"OUTPUT_TILES", caffe2::to_string(output_tile_geometries.tiles)},
                  {"OUTPUT_TILE_X", caffe2::to_string(output_tile_geometries.tile_dims.x)},
                  {"OUTPUT_TILE_WIDTH", caffe2::to_string(output_tile_geometries.tile_size.x)},
                  {"OUTPUT_TILE_HEIGHT", caffe2::to_string(output_tile_geometries.tile_size.y)}}),
        tiling_(tiling) {}

  template <typename T>
  void concat(const GLImageVector<T>** input_images, const GLImageVector<T>& output_image, int size);
  static const char* fragment_shader;
};

// MARK: GLSL

const char* GLConcat::fragment_shader = R"GLSL(#version 300 es
#define TILING                      $(TILING)

// tiling
#define OUTPUT_TILES                $(OUTPUT_TILES)
#define OUTPUT_TILE_X               $(OUTPUT_TILE_X)
#define OUTPUT_TILE_WIDTH           $(OUTPUT_TILE_WIDTH)
#define OUTPUT_TILE_HEIGHT          $(OUTPUT_TILE_HEIGHT)

precision mediump float;
precision mediump int;

in highp vec2 v_texCoord;
TEXTURE_INPUT(inputData);
TEXTURE_OUTPUT(0, outputData);

uniform ivec2 outputSize;
uniform ivec2 inputTileRange; // (]
uniform int input_tile_x;

#if TILING
const ivec2 outputTileSize = ivec2(OUTPUT_TILE_WIDTH, OUTPUT_TILE_HEIGHT);

void main() {
  ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize));
  ivec2 tile = texelCoord / outputTileSize; // 2D output tile idx
  ivec2 tileCoord = texelCoord % outputTileSize; // in-tile coordinates
  int tileNum = OUTPUT_TILE_X * tile.y + tile.x; // 1D output tile idx

  if (tileNum >= inputTileRange.x && tileNum < inputTileRange.y) {
    tileNum = tileNum - inputTileRange.x;
    texelCoord = ivec2(tileNum % input_tile_x, tileNum / input_tile_x)  * ivec2(OUTPUT_TILE_WIDTH, OUTPUT_TILE_HEIGHT) + tileCoord;
    vec4 value = TEXTURE_LOAD(inputData, texelCoord);
    outputData = TEXTURE_STORE(value);
  } else {
    // early termination
    discard;
  }
}

#else
void main() {
  ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize));
  vec4 value = TEXTURE_LOAD(inputData, texelCoord);
  outputData = TEXTURE_STORE(value);
}
#endif

)GLSL";

template <typename T>
void GLConcat::concat(const GLImageVector<T>** input_images, const GLImageVector<T>& output_images, int input_size) {
  for (int k = 0; k < output_images.size(); k++) {
    GLImage<T>* output_image = output_images[k];

    int is = 0, os = 0;
    for (int i = 0; i < input_size; i++) {
      for (int j = 0; j < input_images[i]->slices(); j++) {
        GLImage<T>* input_image = (*input_images[i])[k];
        std::vector<texture_attachment> input_attachments;
        input_attachments.push_back({input_image->textures[j], inputData});

        run(input_attachments,
            {output_image->textures.begin() + os, output_image->textures.begin() + os + 1},
            [&]() {
              glUniform2i(outputSize->location, output_image->texture_width, output_image->texture_height);
              glUniform2i(inputTileRange->location, is, is + input_image->tile_x * input_image->tile_y);
              glUniform1i(input_tile_x->location, input_image->tile_x);
            },
            output_image->texture_width,
            output_image->texture_height);
        if (!tiling_) {
          os++; // for tiling, you always write to the same texture
        }
        is += input_image->tile_x * input_image->tile_y;
      }
    }
  }
}

namespace caffe2 {
template <typename T>
class OpenGLConcatOp final : public Operator<CPUContext>, ImageAllocator<T> {
 public:
  OpenGLConcatOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        order_(StringToStorageOrder(OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW, "OpenGL only supports NCHW order.");
  }

  bool RunOnDevice() override {
    const GLImageVector<T>& input0 = Inputs()[0]->template Get<GLImageVector<T>>();
    const int num_images = input0.size();

    const GLImageVector<T>** input_images = new const GLImageVector<T>*[Inputs().size()];
    input_images[0] = &input0;
    int channelCount = input0.channels();

    bool tiling = OperatorBase::GetSingleArgument<int>("tiling", 0);

    // Only supports input channels divisible by 4 for now
    CAFFE_ENFORCE_EQ(input0.channels() % 4, 0);
    for (auto i = 1; i < Inputs().size(); i++) {
      const GLImageVector<T>& inputi = Inputs()[i]->template Get<GLImageVector<T>>();
      channelCount += inputi.channels();
      CAFFE_ENFORCE_EQ(num_images, inputi.size());
      CAFFE_ENFORCE_EQ(inputi.channels() % 4, 0);
      CAFFE_ENFORCE_EQ(input0.width(), inputi.width());
      CAFFE_ENFORCE_EQ(input0.height(), inputi.height());
      input_images[i] = &inputi;

      if (inputi.tile_x() > 1 || inputi.tile_y() > 1) {
        tiling = true;
      }
    }

    const int input_width = input0.width();
    const int input_height = input0.height();

    const int output_channels = channelCount;
    const int output_width = input_width;
    const int output_height = input_height;

    int output_tile_x = 1;
    int output_tile_y = 1;
    if (tiling) {
      computeOutputTiles(output_channels, output_tile_x, output_tile_y);
    }

    int is_last = OperatorBase::GetSingleArgument<int>("is_last", 0);

    GLImageVector<T>* output = ImageAllocator<T>::newImage(
        num_images, output_width, output_height, output_channels, output_tile_x, output_tile_y, is_last);
    if (!_concat) {
      tile_descriptor output_tile_geometries{
          {output_tile_x, output_tile_y}, {output_width, output_height}, output_tile_x * output_tile_y};
      _concat.reset(new GLConcat(output_tile_geometries, tiling));
    }

    _concat->concat(input_images, *output, Inputs().size());
    delete[] input_images;
    Outputs()[0]->Reset(output);

    return true;
  }

 private:
  StorageOrder order_;
  std::unique_ptr<GLConcat> _concat;
};

REGISTER_CPU_OPERATOR(OpenGLConcat, OpenGLConcatOp<float16_t>);
OPERATOR_SCHEMA(OpenGLConcat).NumInputs(2, 4).NumOutputs(1, 2);
} // namespace caffe2
