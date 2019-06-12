
#include "../core/GLFilter.h"
#include "../core/GLImage.h"

#include "caffe2/core/timer.h"
#include <iostream>
#include <vector>

class GLSoftmaxReduce : public GLFilter {
 public:
  binding* inputTileSize;
  binding* outputSize;
  binding* outputTileSize;
  binding* tileSize;
  binding* spatialTileSize;
  binding* inputTileRange;
  binding* inputData;
  binding* maxData;
  binding* sumData;

  const std::vector<binding*> input_bindings() {
    std::vector<binding*> bindings({BINDING(inputTileSize),
                                    BINDING(outputSize),
                                    BINDING(outputTileSize),
                                    BINDING(tileSize),
                                    BINDING(spatialTileSize),
                                    BINDING(inputTileRange),
                                    BINDING(inputData),
                                    BINDING(maxData),
                                    BINDING(sumData)});
    return bindings;
  }

  GLSoftmaxReduce(
      bool compute_sum_ = false,
      bool tiled = false,
      int input_tile_x = 1)
      : GLFilter(
            "GLSoftmaxReduce",
            vertex_shader,
            fragment_shader,
            input_bindings(),
            {/* no uniform_blocks_bindings */},
            {/* no attributes */},
            {{"COMPUTE_SUM", caffe2::to_string((int)compute_sum_)},
             {"INPUT_TILE_X", caffe2::to_string(input_tile_x)},
             {"TILED_SOFTMAX", caffe2::to_string(int(tiled))}}) {}

  template <typename T>
  void reduce(const GLImage<T>* input_image,
              const GLImage<T>* output_image,
              int tile_size_x,
              int tile_size_y);

  static const char* fragment_shader;
};

// MARK: GLSL

const char* GLSoftmaxReduce::fragment_shader = R"GLSL(#version 300 es

#define TILED_SOFTMAX $(TILED_SOFTMAX)
#define INPUT_TILE_X $(INPUT_TILE_X)
// Compute sum or max
#define COMPUTE_SUM $(COMPUTE_SUM)

precision highp float;
precision mediump int;

in highp vec2 v_texCoord;

uniform ivec2 inputTileSize;
uniform ivec2 outputSize;
uniform ivec2 outputTileSize;
uniform ivec2 spatialTileSize;
uniform ivec2 tileSize;
uniform ivec2 inputTileRange;

TEXTURE_INPUT(inputData);
TEXTURE_OUTPUT(0, outputData);

#if TILED_SOFTMAX
void main() {
  ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize));
  ivec2 tile = texelCoord / outputTileSize; // 2D output tile idx
  ivec2 tileCoord = texelCoord % outputTileSize; // in-tile coordinates
  ivec2 sumArea = min(spatialTileSize, inputTileSize - tileCoord * spatialTileSize);

  vec4 result = vec4(0.0);
  for (int tileIdx = inputTileRange.x; tileIdx < inputTileRange.y; tileIdx++) {
    int inTileX = tileIdx % INPUT_TILE_X;
    int inTileY = tileIdx / INPUT_TILE_X;
    ivec2 inputTileOffset = ivec2(inTileX, inTileY) * inputTileSize;
    for (int y = 0; y < sumArea.y; y++) {
      for (int x = 0; x < sumArea.x; x++) {
        ivec2 idx = tileCoord + ivec2(x, y);
        vec4 val = TEXTURE_LOAD(inputData, inputTileOffset + idx);
  #if COMPUTE_SUM
        result += val;
  #else
        result = max(result, val);
  #endif
      }
    }
  }

  outputData = TEXTURE_STORE(result);
}
#else
void main() {
  ivec2 outputCoord = ivec2(v_texCoord * vec2(outputTileSize));
  ivec2 texelCoord = outputCoord * spatialTileSize;
  ivec2 sumArea = min(spatialTileSize, inputTileSize - texelCoord);
  vec4 result = vec4(0.0);

  for (int y = 0; y < sumArea.y; y++) {
    for (int x = 0; x < sumArea.x; x++) {
      ivec2 idx = texelCoord + ivec2(x, y);
      vec4 val = TEXTURE_LOAD(inputData, idx);
#if COMPUTE_SUM
      result += val;
#else
      result = max(result, val);
#endif
    }
  }

  outputData = TEXTURE_STORE(result);
}
#endif
)GLSL";

template <typename T>
void GLSoftmaxReduce::reduce(const GLImage<T>* input_image,
                             const GLImage<T>* output_image,
                             int tile_size_x,
                             int tile_size_y) {
  int input_slices = input_image->slices;
  int output_slices = output_image->slices;

  for (int is = 0; is < input_slices; is++) {
    std::vector<texture_attachment> input_attachments({{input_image->textures[is], inputData}});
    run(input_attachments,
        {output_image->textures.begin() + is,
         output_image->textures.begin() + is + 1},
        [&]() {
          glUniform2i(
              inputTileSize->location, input_image->width, input_image->height);
          glUniform2i(
              outputSize->location,
              output_image->texture_width,
              output_image->texture_height);
          glUniform2i(
              outputTileSize->location,
              output_image->width,
              output_image->height);
          glUniform2i(
              tileSize->location, input_image->tile_x, input_image->tile_y);
          glUniform2i(spatialTileSize->location, tile_size_x, tile_size_y);
          glUniform2i(
              inputTileRange->location,
              0,
              std::min(
                  (input_image->channels + 3) / 4,
                  input_image->tile_x * input_image->tile_y));
        },
        output_image->texture_width,
        output_image->texture_height);
  }
}

class GLSoftmaxScale : public GLFilter {
 public:
  binding* outputSize;
  binding* inputData;
  binding* maxData;
  binding* sumData;

  const std::vector<binding*> input_bindings() {
    std::vector<binding*> bindings(
        {BINDING(outputSize), BINDING(inputData), BINDING(maxData), BINDING(sumData)});
    return bindings;
  }

  GLSoftmaxScale(bool _compute_exp = false, bool tiled = false)
      : GLFilter(
            "GLSoftmaxScale",
            vertex_shader,
            fragment_shader,
            input_bindings(),
            {/* no uniform blocks */},
            {/* no attributes */},
            {{"COMPUTE_EXP", caffe2::to_string((int)_compute_exp)},
             {"TILED_SOFTMAX", caffe2::to_string((int)tiled)}}) {}

  template <typename T>
  void scale(const GLImage<T>* input_image,
             const GLImage<T>* max_image,
             const GLImage<T>* sum_image,
             const GLImage<T>* output_image);

  static const char* fragment_shader;
};

template <typename T>
void GLSoftmaxScale::scale(const GLImage<T>* input_image,
                           const GLImage<T>* max_image,
                           const GLImage<T>* sum_image,
                           const GLImage<T>* output_image) {
  int input_slices = input_image->slices;
  int output_slices = output_image->slices;

  for (int is = 0; is < input_slices; is++) {
    std::vector<texture_attachment> input_attachments({{input_image->textures[is], inputData},
                                                       {max_image->textures[is], maxData},
                                                       {sum_image->textures[is], sumData}});
    run(input_attachments,
        {output_image->textures.begin() + is,
         output_image->textures.begin() + is + 1},
        [&]() {
          glUniform2i(
              outputSize->location,
              output_image->texture_width,
              output_image->texture_height);
        },
        output_image->texture_width,
        output_image->texture_height);
  }
}

// MARK: GLSL

const char* GLSoftmaxScale::fragment_shader = R"GLSL(#version 300 es

#define COMPUTE_EXP $(COMPUTE_EXP)
#define TILED_SOFTMAX $(TILED_SOFTMAX)

precision highp float;
precision mediump int;

in highp vec2 v_texCoord;
uniform ivec2 outputSize;

TEXTURE_INPUT(inputData);
TEXTURE_INPUT(maxData);
TEXTURE_INPUT(sumData);
TEXTURE_OUTPUT(0, outputData);

void main() {
  ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize));
  vec4 val = TEXTURE_LOAD(inputData, texelCoord);
#if COMPUTE_EXP
  vec4 maxVal = TEXTURE_LOAD(maxData, ivec2(0));
  #if TILED_SOFTMAX
    float singleMax = max(max(max(maxVal.x, maxVal.y), maxVal.z), maxVal.w);
    maxVal = vec4(singleMax, singleMax, singleMax, singleMax);
    outputData = TEXTURE_STORE(exp(val - maxVal));
  #else
    outputData = TEXTURE_STORE(exp(val - maxVal));
  #endif

#else
  vec4 sumVal = TEXTURE_LOAD(sumData, ivec2(0));
  #if TILED_SOFTMAX
    float singleSum = sumVal.x + sumVal.y + sumVal.z + sumVal.w;
    sumVal = vec4(singleSum, singleSum, singleSum, singleSum);
    outputData = TEXTURE_STORE(val / sumVal);
  #else
    outputData = TEXTURE_STORE(val / sumVal);
  #endif
#endif

}
)GLSL";

#include "../core/ImageAllocator.h"
#include "caffe2/core/operator.h"

#ifndef CAFFE2_MOBILE
#error "Caffe2 mobile state not defined"
#endif

#if CAFFE2_MOBILE

namespace caffe2 {
template <class T>
class OpenGLSoftmax final : public Operator<CPUContext>, ImageAllocator<T> {
 public:
  OpenGLSoftmax(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        order_(StringToStorageOrder(OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW, "OpenGL only supports NCHW order.");
  }

  bool RunOnDevice() override {
    const GLImageVector<T>& input = Inputs()[INPUT]->template Get<GLImageVector<T>>();
    const int num_images = input.size();
    const int input_channels = input.channels();
    const int input_width = input.width();
    const int input_height = input.height();

    const int output_channels = input_channels;
    const int output_width = input_width;
    const int output_height = input_height;

    int is_last = OperatorBase::GetSingleArgument<int>("is_last", 0);
    // For tiling
    const int input_tile_x = input.tile_x(), input_tile_y = input.tile_y();
    const int output_tile_x = input_tile_x, output_tile_y = input_tile_y;
    const bool tiled = input_tile_x > 1 || input_tile_y > 1;
    if (tiled) {
      CAFFE_ENFORCE_EQ(
          input.slices(), 1, "Input needs to be tiled in a single texture");
    }

    CAFFE_ENFORCE(
        tiled || input_channels == 1,
        "Softmax only works for input_channel == 1 or input_channel > 1 with tiling enabled.");

    // for spatial dimension
    const int tile_size_x = 16;
    const int tile_size_y = 16;

    int max_buf_width = input_width;
    int max_buf_height = input_height;
    int max_buf_channels = input_channels;
    vector<GLImageVector<T>*> reduce_buf;

    while (reduce_buf.size() == 0 || (max_buf_height > tile_size_y)) {
      max_buf_width = (max_buf_width + tile_size_x - 1) / tile_size_x;
      max_buf_height = (max_buf_height + tile_size_y - 1) / tile_size_y;
      if (tiled) {
        // since we are summing over all the channels within a channel tile
        max_buf_channels =
            (max_buf_channels + input_tile_x * input_tile_y - 1) /
            (input_tile_x + input_tile_y);
      }
      reduce_buf.push_back(ImageAllocator<T>::newImage(
          1,
          max_buf_width,
          max_buf_height,
          max_buf_channels,
          output_tile_x,
          output_tile_y));
    }

    GLImageVector<T>* max = ImageAllocator<T>::newImage(num_images, 1, 1, 1);
    GLImageVector<T>* sum = ImageAllocator<T>::newImage(num_images, 1, 1, 1);
    GLImageVector<T>* after_exp = ImageAllocator<T>::newImage(
        num_images,
        output_width,
        output_height,
        output_channels,
        output_tile_x,
        output_tile_y);
    GLImageVector<T>* output_images = ImageAllocator<T>::newImage(
        num_images,
        output_width,
        output_height,
        output_channels,
        output_tile_x,
        output_tile_y,
        is_last);

    if (!f_max) {
      f_max.reset(new GLSoftmaxReduce(false, tiled, input_tile_x));
      f_exp.reset(new GLSoftmaxScale(true, tiled));
      f_sum.reset(new GLSoftmaxReduce(true, tiled, input_tile_x));
      f_scale.reset(new GLSoftmaxScale(false, tiled));
    }

    for (int i = 0; i < num_images; i++) {
      auto input_image = input[i];
      auto max_image = (*max)[i];
      auto sum_image = (*sum)[i];
      auto after_exp_image = (*after_exp)[i];
      auto output_image = (*output_images)[i];
      // Get Max
      for (int ir = 0; ir < reduce_buf.size() + 1; ir++) {
        const GLImage<T>* in = ir == 0 ? input_image : (*reduce_buf[ir - 1])[0];
        GLImage<T>* out = ir == reduce_buf.size() ? max_image : (*reduce_buf[ir])[0];

        const int running_tile_size_x =
            ir < reduce_buf.size() ? tile_size_x : in->width;
        const int running_tile_size_y =
            ir < reduce_buf.size() ? tile_size_y : in->height;
        f_max->reduce(in, out, running_tile_size_x, running_tile_size_y);
      }
      // scale vals by exp(x - max)
      f_exp->scale(input_image, max_image, sum_image, after_exp_image);

      // Get sum of the exp
      for (int ir = 0; ir < reduce_buf.size() + 1; ir++) {
        const GLImage<T>* in = ir == 0 ? after_exp_image : (*reduce_buf[ir - 1])[0];
        GLImage<T>* out = ir == reduce_buf.size() ? sum_image : (*reduce_buf[ir])[0];
        const int running_tile_size_x = ir < reduce_buf.size() ? tile_size_x : in->width;
        const int running_tile_size_y = ir < reduce_buf.size() ? tile_size_y : in->height;
        f_sum->reduce(in, out, running_tile_size_x, running_tile_size_y);
      }

      // Scale(softmax)
      f_scale->scale(after_exp_image, max_image, sum_image, output_image);
    }

    Outputs()[OUTPUT]->Reset(output_images);

    delete sum;
    delete max;
    delete after_exp;
    for (auto&& rb : reduce_buf) {
      delete rb;
    }
    return true;
  }

 private:
  StorageOrder order_;
  std::unique_ptr<GLSoftmaxReduce> f_max;
  std::unique_ptr<GLSoftmaxScale> f_exp;
  std::unique_ptr<GLSoftmaxReduce> f_sum;
  std::unique_ptr<GLSoftmaxScale> f_scale;

  INPUT_TAGS(INPUT, FILTER, BIAS);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_CPU_OPERATOR(OpenGLSoftmax, OpenGLSoftmax<float16_t>);
OPERATOR_SCHEMA(OpenGLSoftmax)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape();
} // namespace caffe2
#endif // CAFFE2_MOBILE
