
#include "../core/GLFilter.h"
#include "../core/GLImage.h"
#include "../core/ImageAllocator.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include <iostream>
#include <vector>

class GLReduce : public GLFilter {
 public:
  binding* inputSize;
  binding* outputSize;
  binding* tileSize;
  binding* inv_pixel_count;
  binding* epsilon;
  binding* inputData;
  binding* averageData;

  bool compute_inv_stdev;
  bool compute_norm;

  const std::vector<binding*> input_bindings(bool compute_norm_) {
    std::vector<binding*> bindings({BINDING(inputSize),
                                    BINDING(outputSize),
                                    BINDING(tileSize),
                                    BINDING(inv_pixel_count),
                                    BINDING(epsilon),
                                    BINDING(inputData)});
    if (compute_norm_) {
      bindings.push_back(BINDING(averageData));
    }
    return bindings;
  }

  GLReduce(bool compute_inv_stdev_ = false, bool compute_norm_ = false)
      : GLFilter("GLReduce",
                 vertex_shader,
                 fragment_shader,
                 input_bindings(compute_norm_),
                 {/* no uniform_blocks_bindings */},
                 {/* no attributes */},
                 {{"COMPUTE_INV_STDEV", caffe2::to_string((int)compute_inv_stdev_)},
                  {"COMPUTE_NORM", caffe2::to_string((int)compute_norm_)}}),
        compute_inv_stdev(compute_inv_stdev_),
        compute_norm(compute_norm_) {}

  template <typename T>
  void reduce(const GLImage<T>* input_image,
              const GLImage<T>* output_image,
              int tile_size_x,
              int tile_size_y,
              float inv_pixel_count_ = 1.0,
              float epsilon_ = 0.0);

  template <typename T>
  void norm(const GLImage<T>* input_image,
            const GLImage<T>* avg_image,
            const GLImage<T>* output_image,
            int tile_size_x,
            int tile_size_y,
            float inv_pixel_count_);

  static const char* fragment_shader;
};

// MARK: GLSL

const char* GLReduce::fragment_shader = R"GLSL(#version 300 es

#define COMPUTE_INV_STDEV $(COMPUTE_INV_STDEV)
#define COMPUTE_NORM $(COMPUTE_NORM)

precision mediump float;
precision mediump int;

in highp vec2 v_texCoord;

uniform ivec2 inputSize;
uniform ivec2 outputSize;
uniform ivec2 tileSize;
uniform float inv_pixel_count;
uniform float epsilon;

#if COMPUTE_NORM
TEXTURE_INPUT(averageData);
#endif

TEXTURE_INPUT(inputData);
TEXTURE_OUTPUT(0, outputData);

void main() {
  ivec2 outputCoord = ivec2(v_texCoord * vec2(outputSize));
  ivec2 texelCoord = outputCoord * tileSize;
  ivec2 sumArea = min(tileSize, inputSize - texelCoord);
  highp vec4 sum = vec4(0.0);

#if COMPUTE_NORM
  vec4 avg = TEXTURE_LOAD(averageData, ivec2(0));
#endif

  for (int y = 0; y < sumArea.y; y++) {
    for (int x = 0; x < sumArea.x; x++) {
      ivec2 idx = texelCoord + ivec2(x, y);
      vec4 val = TEXTURE_LOAD(inputData, idx);
#if COMPUTE_NORM
      val -= avg;
      sum += val * val;
#else
      sum += val;
#endif
    }
  }

#if COMPUTE_INV_STDEV
  outputData = TEXTURE_STORE(inversesqrt(sum * vec4(inv_pixel_count) + vec4(epsilon)));
#elif COMPUTE_NORM
  outputData = TEXTURE_STORE(sum * vec4(inv_pixel_count));
#else
  outputData = TEXTURE_STORE(sum * vec4(inv_pixel_count) + vec4(epsilon));
#endif
}

)GLSL";

template <typename T>
void GLReduce::reduce(const GLImage<T>* input_image,
                      const GLImage<T>* output_image,
                      int tile_size_x,
                      int tile_size_y,
                      float inv_pixel_count_,
                      float epsilon_) {
  int input_slices = input_image->slices;
  int output_slices = output_image->slices;

  for (int is = 0; is < input_slices; is++) {
    std::vector<texture_attachment> input_attachments({{input_image->textures[is], inputData}});

    run(input_attachments,
        {output_image->textures.begin() + is, output_image->textures.begin() + is + 1},
        [&]() {
          glUniform2i(inputSize->location, input_image->width, input_image->height);
          glUniform2i(outputSize->location, output_image->width, output_image->height);
          glUniform2i(tileSize->location, tile_size_x, tile_size_y);
          glUniform1f(inv_pixel_count->location, inv_pixel_count_);
          glUniform1f(epsilon->location, epsilon_);
        },
        output_image->width,
        output_image->height);
  }
}

template <typename T>
void GLReduce::norm(const GLImage<T>* input_image,
                    const GLImage<T>* avg_image,
                    const GLImage<T>* output_image,
                    int tile_size_x,
                    int tile_size_y,
                    float inv_pixel_count_) {
  int input_slices = input_image->slices;
  int output_slices = output_image->slices;

  for (int is = 0; is < input_slices; is++) {
    std::vector<texture_attachment> input_attachments(
        {{input_image->textures[is], inputData}, {avg_image->textures[is], averageData}});

    run(input_attachments,
        {output_image->textures.begin() + is, output_image->textures.begin() + is + 1},
        [&]() {
          glUniform2i(inputSize->location, input_image->width, input_image->height);
          glUniform2i(outputSize->location, output_image->width, output_image->height);
          glUniform2i(tileSize->location, tile_size_x, tile_size_y);
          glUniform1f(inv_pixel_count->location, inv_pixel_count_);
        },
        output_image->width,
        output_image->height);
  }
}

class GLScale : public GLFilter {
 public:
  binding* outputSize;
  binding* inputData;
  binding* averageData;
  binding* normData;

  binding* scale_factor;
  binding* bias_factor;
  binding* prelu_scale_factor;

  const int channels;
  const float* scale;
  const float* bias;
  const float* prelu_scale;
  const int prelu_size;

  const std::vector<binding*> input_bindings(bool fuse_prelu) {
    std::vector<binding*> bindings({BINDING(outputSize),
                                    BINDING(scale_factor),
                                    BINDING(bias_factor),
                                    BINDING(inputData),
                                    BINDING(averageData),
                                    BINDING(normData)});
    if (fuse_prelu) {
      bindings.push_back(prelu_scale_factor = new binding({"prelu_scale_factor"}));
    }
    return bindings;
  }

  GLScale(const int _channels,
          const float* _scale,
          const float* _bias,
          const float* _prelu_scale = nullptr,
          const int _prelu_size = 0)
      : GLFilter("GLScale",
                 vertex_shader,
                 fragment_shader,
                 input_bindings(_prelu_scale != nullptr),
                 {/* no uniform blocks */},
                 {/* no attributes */},
                 {{"FUSE_PRELU", caffe2::to_string(_prelu_scale != nullptr)}}),
        channels(_channels),
        scale(_scale),
        bias(_bias),
        prelu_scale(_prelu_scale),
        prelu_size(_prelu_size) {}

  template <typename T>
  void scale_and_shift(const GLImage<T>* input_image,
                       const GLImage<T>* avg_image,
                       const GLImage<T>* norm_image,
                       const GLImage<T>* output_image);

  static const char* fragment_shader;
};

// MARK: GLSL

const char* GLScale::fragment_shader = R"GLSL(#version 300 es

#define FUSE_PRELU $(FUSE_PRELU)

precision mediump float;
precision mediump int;

in highp vec2 v_texCoord;
uniform ivec2 outputSize;
uniform vec4 scale_factor;
uniform vec4 bias_factor;

#if FUSE_PRELU
uniform vec4 prelu_scale_factor;
#endif

TEXTURE_INPUT(inputData);
TEXTURE_INPUT(averageData);
TEXTURE_INPUT(normData);
TEXTURE_OUTPUT(0, outputData);

void main() {
  ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize));

  vec4 val = TEXTURE_LOAD(inputData, texelCoord);
  vec4 avg = TEXTURE_LOAD(averageData, ivec2(0));
  vec4 inv_stdev = TEXTURE_LOAD(normData, ivec2(0));

#if FUSE_PRELU
  vec4 result = (val - avg) * inv_stdev * scale_factor + bias_factor;
  vec4 o = mix(result * prelu_scale_factor, result, vec4(greaterThan(result, vec4(0))));
  outputData = TEXTURE_STORE(o);
#else
  vec4 o = (val - avg) * inv_stdev * scale_factor + bias_factor;
  outputData = TEXTURE_STORE(o);
#endif
}

)GLSL";

template <typename T>
void GLScale::scale_and_shift(const GLImage<T>* input_image,
                              const GLImage<T>* avg_image,
                              const GLImage<T>* norm_image,
                              const GLImage<T>* output_image) {
  int input_slices = input_image->slices;
  int output_slices = output_image->slices;

  for (int is = 0; is < input_slices; is++) {
    std::vector<texture_attachment> input_attachments({{input_image->textures[is], inputData},
                                                       {avg_image->textures[is], averageData},
                                                       {norm_image->textures[is], normData}});

    run(input_attachments,
        {output_image->textures.begin() + is, output_image->textures.begin() + is + 1},
        [&]() {
          glUniform2i(outputSize->location, output_image->width, output_image->height);
          glUniform4f(scale_factor->location,
                      scale[4 * is],
                      channels > 4 * is + 1 ? scale[4 * is + 1] : 0,
                      channels > 4 * is + 2 ? scale[4 * is + 2] : 0,
                      channels > 4 * is + 3 ? scale[4 * is + 3] : 0);
          glUniform4f(bias_factor->location,
                      bias[4 * is],
                      channels > 4 * is + 1 ? bias[4 * is + 1] : 0,
                      channels > 4 * is + 2 ? bias[4 * is + 2] : 0,
                      channels > 4 * is + 3 ? bias[4 * is + 3] : 0);
          if (prelu_scale != nullptr) {
            glUniform4f(prelu_scale_factor->location,
                        prelu_size == channels ? prelu_scale[4 * is] : prelu_scale[0],
                        channels > 4 * is + 1 && prelu_size == channels ? prelu_scale[4 * is + 1]
                                                                        : prelu_scale[0],
                        channels > 4 * is + 2 && prelu_size == channels ? prelu_scale[4 * is + 2]
                                                                        : prelu_scale[0],
                        channels > 4 * is + 3 && prelu_size == channels ? prelu_scale[4 * is + 3]
                                                                        : prelu_scale[0]);
          }
        },
        output_image->width,
        output_image->height);
  }
}

namespace caffe2 {
template <class T, bool FUSE_PRELU>
class OpenGLInstanceNormPReluOp final : public Operator<CPUContext>, ImageAllocator<T> {
 public:
  OpenGLInstanceNormPReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5)),
        order_(StringToStorageOrder(OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE(epsilon_ >= 0, "Must pass a nonnegative epsilon.");
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW, "Metal only supports NCHW order.");
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

    const int tile_size_x = 16;
    const int tile_size_y = 16;
    int avg_buf_width = input_width;
    int avg_buf_height = input_height;

    vector<GLImageVector<T>*> reduce_buf;
    while (reduce_buf.size() == 0 ||
           (avg_buf_width > tile_size_x && avg_buf_height > tile_size_y)) {
      avg_buf_width = (avg_buf_width + tile_size_x - 1) / tile_size_x;
      avg_buf_height = (avg_buf_height + tile_size_y - 1) / tile_size_y;

      reduce_buf.push_back(
          ImageAllocator<T>::newImage(1, avg_buf_width, avg_buf_height, output_channels));
    }

    GLImageVector<T>* avg = ImageAllocator<T>::newImage(num_images, 1, 1, output_channels);
    GLImageVector<T>* inv_stdev = ImageAllocator<T>::newImage(num_images, 1, 1, output_channels);
    GLImageVector<T>* output = ImageAllocator<T>::newImage(
        num_images, output_width, output_height, output_channels, is_last);
    const float* prelu_data = nullptr;
    int prelu_size = 0;
    if (FUSE_PRELU) {
      DCHECK_EQ(InputSize(), 4);
      const auto& prelu_scale = Input(PRELU);
      prelu_data = prelu_scale.template data<float>();
      prelu_size = prelu_scale.size();
    } else {
      DCHECK_EQ(InputSize(), 3);
    }

    const auto& scale = Input(SCALE);
    const auto& bias = Input(BIAS);

    if (!f_reduce) {
      f_reduce.reset(new GLReduce());
      f_norm.reset(new GLReduce(false, true));
      f_stdDev.reset(new GLReduce(true, false));
      f_scale.reset(new GLScale(input_channels,
                                scale.template data<float>(),
                                bias.template data<float>(),
                                prelu_data,
                                prelu_size));
    }

    for (int i = 0; i < num_images; i++) {
      for (int k = 0; k < reduce_buf.size() + 1; k++) {
        const GLImage<T>* in = k == 0 ? input[i] : (*reduce_buf[k - 1])[0];
        GLImage<T>* out = k == reduce_buf.size() ? (*avg)[i] : (*reduce_buf[k])[0];

        float norm = k < reduce_buf.size()
                         ? 1.0 / (tile_size_x * tile_size_y)
                         : (float)pow(tile_size_x * tile_size_y, reduce_buf.size()) /
                               (float)(input_width * input_height);
        const int running_tile_size_x = k < reduce_buf.size() ? tile_size_x : in->width;
        const int running_tile_size_y = k < reduce_buf.size() ? tile_size_y : in->height;
        f_reduce->reduce(in, out, running_tile_size_x, running_tile_size_y, norm);
      }

      for (int k = 0; k < reduce_buf.size() + 1; k++) {
        const GLImage<T>* in = k == 0 ? input[i] : (*reduce_buf[k - 1])[0];
        GLImage<T>* out = k == reduce_buf.size() ? (*inv_stdev)[i] : (*reduce_buf[k])[0];

        float norm = k < reduce_buf.size()
                         ? 1.0 / (tile_size_x * tile_size_y)
                         : (float)pow(tile_size_x * tile_size_y, reduce_buf.size()) /
                               (float)(input_width * input_height);

        if (k == 0) {
          f_norm->norm(in, (*avg)[i], out, tile_size_x, tile_size_y, norm);
        } else if (k < reduce_buf.size()) {
          f_reduce->reduce(in, out, tile_size_x, tile_size_y, norm);
        } else {
          const int running_tile_size_x = k < reduce_buf.size() ? tile_size_x : in->width;
          const int running_tile_size_y = k < reduce_buf.size() ? tile_size_y : in->height;
          f_stdDev->reduce(in, out, running_tile_size_x, running_tile_size_y, norm, epsilon_);
        }
      }

      f_scale->scale_and_shift(input[i], (*avg)[i], (*inv_stdev)[i], (*output)[i]);
    }
    Outputs()[OUTPUT]->Reset(output);
    if (OutputSize() > 1) {
      Outputs()[MEAN]->Reset(avg);
      Outputs()[INV_STDEV]->Reset(inv_stdev);
    } else {
      delete avg;
      delete inv_stdev;
    }
    for (auto&& rb : reduce_buf) {
      delete rb;
    }

    return true;
  }

 private:
  float epsilon_;
  StorageOrder order_;
  std::unique_ptr<GLReduce> f_reduce;
  std::unique_ptr<GLReduce> f_norm;
  std::unique_ptr<GLReduce> f_stdDev;
  std::unique_ptr<GLScale> f_scale;

  INPUT_TAGS(INPUT, SCALE, BIAS, PRELU);
  OUTPUT_TAGS(OUTPUT, MEAN, INV_STDEV);
};

REGISTER_CPU_OPERATOR(OpenGLInstanceNorm, OpenGLInstanceNormPReluOp<float16_t, false>);
OPERATOR_SCHEMA(OpenGLInstanceNorm).NumInputs(3, 4).NumOutputs(1, 3).AllowInplace({{0, 0}});
REGISTER_CPU_OPERATOR(OpenGLInstanceNormPRelu, OpenGLInstanceNormPReluOp<float16_t, true>);
OPERATOR_SCHEMA(OpenGLInstanceNormPRelu).NumInputs(3, 4).NumOutputs(1, 3).AllowInplace({{0, 0}});
} // namespace caffe2
