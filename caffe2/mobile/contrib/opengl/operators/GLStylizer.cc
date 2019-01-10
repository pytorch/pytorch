
#include "../core/GLContext.h"
#include "../core/GLFilter.h"
#include "../core/GLImage.h"
#include "../core/ImageAllocator.h"

#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

enum InputFormat { BGRA = 0, RGBA = 1 };

class GLStylizer : public GLFilter {
  binding* inputData;
  binding* outputSize;
  binding* mean;
  binding* noise_std;
  bool deprocess;

 public:
  GLStylizer(bool _deprocess = false, InputFormat input_format = BGRA)
      : GLFilter(_deprocess ? "GLDeStylizer" : "GLStylizer",
                 vertex_shader,
                 fragment_shader,
                 std::vector<binding*>({BINDING(inputData), BINDING(mean), BINDING(noise_std), BINDING(outputSize)}),
                 {/* no uniform blocks */},
                 {/* no attributes */},
                 {{"DEPROCESS", caffe2::to_string(_deprocess)}, {"RGBAINPUT", caffe2::to_string(input_format)}}),
        deprocess(_deprocess) {}

  template <typename T1, typename T2>
  void stylize(const GLImage<T1>* input_image,
               const GLImage<T2>* output_image,
               const float mean_values[3],
               float noise_std_value);

  static const char* fragment_shader;
};

// MARK: GLSL

const char* GLStylizer::fragment_shader = R"GLSL(#version 300 es

#define DEPROCESS         $(DEPROCESS)
#define RGBAINPUT         $(RGBAINPUT)

precision mediump float;
precision mediump int;
precision mediump sampler2D;

in highp vec2 v_texCoord;

uniform ivec2 outputSize;

uniform vec3 mean;
uniform float noise_std;

#if DEPROCESS
TEXTURE_INPUT(inputData);
layout(location = 0) out mediump vec4 outputData;
#else
uniform sampler2D inputData;
TEXTURE_OUTPUT(0, outputData);
#endif

#if !DEPROCESS
// http://byteblacksmith.com/improvements-to-the-canonical-one-liner-glsl-rand-for-opengl-es-2-0/

highp float rand(vec2 co) {
  highp float a = 12.9898;
  highp float b = 78.233;
  highp float c = 43758.5453;
  highp float dt = dot(co.xy, vec2(a, b));
  highp float sn = mod(dt, 3.14);
  return fract(sin(sn) * c);
}
#endif

// In AR Engine, input/output a RBGA texture; otherwise, BGRA tensor => texture
#if RGBAINPUT
void main() {
#if DEPROCESS
  ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize));
  vec4 val = TEXTURE_LOAD(inputData, texelCoord);
  outputData = vec4((val.rgb + mean) / 255.0, 1.0).bgra;
#else
  outputData = TEXTURE_STORE(vec4(255.0 * texture(inputData, v_texCoord).bgr - mean + vec3(noise_std * rand(v_texCoord)), 0.0));
#endif
}
#else
void main() {
#if DEPROCESS
  ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize));
  vec4 val = TEXTURE_LOAD(inputData, texelCoord);
  outputData = vec4((val.rgb + mean) / 255.0, 1.0);
#else
  outputData = TEXTURE_STORE(vec4(255.0 * texture(inputData, v_texCoord).rgb - mean + vec3(noise_std * rand(v_texCoord)), 0.0));
#endif
}
#endif
)GLSL";

template <typename T1, typename T2>
void GLStylizer::stylize(const GLImage<T1>* input_image,
                         const GLImage<T2>* output_image,
                         const float mean_values[3],
                         float noise_std_value) {
  int input_slices = input_image->slices;
  int output_slices = output_image->slices;

  run(std::vector<texture_attachment>({{input_image->textures[0], inputData}}),
      {output_image->textures[0]},
      [&]() {
        glUniform2i(outputSize->location, output_image->width, output_image->height);
        glUniform3f(mean->location, mean_values[0], mean_values[1], mean_values[2]);
        if (!deprocess) {
          glUniform1f(noise_std->location, noise_std_value);
        }
      },
      output_image->width,
      output_image->height);
}

namespace caffe2 {
class OpenGLTensorToTextureStylizerPreprocessOp : public Operator<CPUContext>,
                                                  ImageAllocator<uint8_t>,
                                                  ImageAllocator<float16_t> {
 public:
  // Expect this many channels as input
  static constexpr int kInputChannels = 4;

  // Expect this many channels as output
  static constexpr int kOutputChannels = 3;

  USE_OPERATOR_BASE_FUNCTIONS;

  OpenGLTensorToTextureStylizerPreprocessOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() {
    const auto& input = Input(0);
    const auto& mean = Input(1);

    CAFFE_ENFORCE(input.ndim() == 4);

    const int num_images = input.dim32(0);
    const int input_height = input.dim32(1);
    const int input_width = input.dim32(2);
    const int input_channels = input.dim32(3);

    CAFFE_ENFORCE(input.dim32(0) == 1); // N == 1
    CAFFE_ENFORCE(input_channels == kInputChannels);
    CAFFE_ENFORCE(mean.size() == kOutputChannels); // Assume BGR or BGRA

    // get the buffers from input tensors
    const float* mean_buffer = mean.template data<float>();
    const uint8_t* input_buffer = input.template data<uint8_t>();

    // set up the OpenGL context
    GLContext::getGLContext()->set_context();

    GLImageVector<float16_t>* output_images = ImageAllocator<float16_t>::newImage(num_images,
                                                                                  input_width,
                                                                                  input_height,
                                                                                  kOutputChannels,
#if CAFFE2_IOS
                                                                                  true
#else
                                                                                  false
#endif
    );
    const int tile_x = 1, tile_y = 1;
    GLImageVector<uint8_t>* input_images = ImageAllocator<uint8_t>::newImage(
        num_images, input_width, input_height, kInputChannels, tile_x, tile_y, false);
    for (int i = 0; i < num_images; i++) {
      auto input_image = (*input_images)[i];
      auto output_image = (*output_images)[i];
      const GLTexture* inputTexture = input_image->textures[0];
      inputTexture->loadData(input_buffer);

      if (!glStylizer_) {
        glStylizer_.reset(new GLStylizer());
      }

      glStylizer_->stylize(
          input_image, output_image, mean_buffer, GetSingleArgument<float>("noise_std", 10.0));
    }
    delete input_images;
    Outputs()[0]->Reset(output_images);

    return true;
  }

 private:
  std::unique_ptr<GLStylizer> glStylizer_;
};

template <InputFormat inputFormat>
class OpenGLTextureToTextureStylizerPreprocessOp : public Operator<CPUContext>,
                                                   ImageAllocator<uint8_t>,
                                                   ImageAllocator<float16_t> {
 public:
  // Expect this many channels as input
  static constexpr int kInputChannels = 4;

  // Expect this many channels as output
  static constexpr int kOutputChannels = 3;

  USE_OPERATOR_BASE_FUNCTIONS;

  OpenGLTextureToTextureStylizerPreprocessOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() {
    const GLImageVector<uint8_t>& input = Inputs()[0]->template Get<GLImageVector<uint8_t>>();
    const auto& mean = Input(1);

    const int num_images = input.size();
    const int input_height = input.height();
    const int input_width = input.width();
    const int input_channels = input.channels();

    CAFFE_ENFORCE_GT(num_images, 0);
    CAFFE_ENFORCE(input[0]->slices == 1); // N == 1
    CAFFE_ENFORCE(input_channels == kInputChannels);
    CAFFE_ENFORCE(mean.size() == kOutputChannels); // Assume BGR or BGRA

    // get the buffers from input tensors
    const float* mean_buffer = mean.template data<float>();

    GLImageVector<float16_t>* output_images = ImageAllocator<float16_t>::newImage(
        num_images, input_width, input_height, kOutputChannels, false);

    if (!glStylizer_) {
      glStylizer_.reset(new GLStylizer(false, inputFormat));
    }
    for (int i = 0; i < num_images; i++) {
      auto input_image = input[i];
      auto output_image = (*output_images)[i];
      glStylizer_->stylize(
          input_image, output_image, mean_buffer, GetSingleArgument<float>("noise_std", 10.0));
    }
    Outputs()[0]->Reset(output_images);

    return true;
  }

 private:
  std::unique_ptr<GLStylizer> glStylizer_;
};

REGISTER_CPU_OPERATOR(OpenGLTensorToTextureStylizerPreprocess,
                      OpenGLTensorToTextureStylizerPreprocessOp);
OPERATOR_SCHEMA(OpenGLTensorToTextureStylizerPreprocess).NumInputs(2).NumOutputs(1);

REGISTER_CPU_OPERATOR(OpenGLTextureToTextureStylizerPreprocess,
                      OpenGLTextureToTextureStylizerPreprocessOp<RGBA>);
OPERATOR_SCHEMA(OpenGLTextureToTextureStylizerPreprocess).NumInputs(2).NumOutputs(1);

class OpenGLTextureToTensorStylizerDeprocessOp : public Operator<CPUContext>,
                                                 ImageAllocator<uint8_t> {
 public:
  using Operator<CPUContext>::Operator;

  // Expect this many channels as input
  static constexpr int kInputChannels = 3;

  // Expect this many channels as output
  static constexpr int kOutputChannels = 4;

  bool RunOnDevice() {
    const GLImageVector<float16_t>& input = Inputs()[0]->template Get<GLImageVector<float16_t>>();
    const auto& mean = Input(1);
    auto* output = Output(0);

    const int num_images = input.size(), channels = input.channels(), height = input.height(),
              width = input.width();
    // Assume BGR or BGRA
    CAFFE_ENFORCE(mean.size() == kInputChannels);
    CAFFE_ENFORCE(channels == kInputChannels);
    // RGB
    output->Resize(num_images, height, width, kOutputChannels);

    const auto* mean_data = mean.template data<float>();
    auto* output_buffer = output->template mutable_data<uint8_t>();

    GLImageVector<uint8_t>* output_images =
        ImageAllocator<uint8_t>::newImage(num_images, width, height, kOutputChannels, true);

    if (!glStylizer_) {
      glStylizer_.reset(new GLStylizer(true));
    }

    for (int i = 0; i < num_images; i++) {
      auto input_image = input[i];
      auto output_image = (*output_images)[i];
      glStylizer_->stylize(input_image, output_image, mean_data, 0);

      output_image->textures[0]->map_read([&](const void* buffer,
                                              size_t width,
                                              size_t height,
                                              size_t stride,
                                              size_t channels,
                                              const GLTexture::Type& type) {
        if (width == stride) {
          memcpy(output_buffer, buffer, channels * width * height);
        } else {
          typedef uint8_t(input_data_t)[height][stride][channels];
          typedef uint8_t(output_data_t)[height][width][channels];

          const input_data_t& input_data = *reinterpret_cast<const input_data_t*>(buffer);
          output_data_t& output_data = *reinterpret_cast<output_data_t*>(output_buffer);

          for (int y = 0; y < height; y++) {
            memcpy(output_data[y], input_data[y], channels * width);
          }
        }
      });
    }
    delete output_images;

    return true;
  }

 private:
  std::unique_ptr<GLStylizer> glStylizer_;
};

template <InputFormat inputFormat>
class OpenGLTextureToTextureStylizerDeprocessOp : public Operator<CPUContext>,
                                                  ImageAllocator<uint8_t> {
 public:
  using Operator<CPUContext>::Operator;

  // Expect this many channels as input
  static constexpr int kInputChannels = 3;

  // Expect this many channels as output
  static constexpr int kOutputChannels = 4;

  bool RunOnDevice() {
    const GLImageVector<float16_t>& input = Inputs()[0]->template Get<GLImageVector<float16_t>>();
    const auto& mean = Input(1);

    const int num_images = input.size(), channels = input.channels(), height = input.height(),
              width = input.width();

    CAFFE_ENFORCE(mean.size() == kInputChannels);
    CAFFE_ENFORCE(channels == kInputChannels);

    const auto* mean_data = mean.template data<float>();

    // Use foreignTextureAllocator inside GLContext
    // glDeleteTexture will not be called from inside caffe2 for this texture
    GLImageVector<uint8_t>* output_images;
    auto textureAllocator = GLContext::getGLContext()->getTextureAllocator();
    const int tile_x = 1, tile_y = 1;
    if (textureAllocator != nullptr) {
      output_images = ImageAllocator<uint8_t>::newImage(
          num_images, width, height, kOutputChannels, tile_x, tile_y, textureAllocator);
    } else {
      // fallback when textureAllocator is not set
      output_images = ImageAllocator<uint8_t>::newImage(num_images, width, height, kOutputChannels);
    }

    if (!glStylizer_) {
      glStylizer_.reset(new GLStylizer(true, inputFormat));
    }

    for (int i = 0; i < num_images; i++) {
      auto input_image = input[i];
      auto output_image = (*output_images)[i];
      glStylizer_->stylize(input_image, output_image, mean_data, 0);
    }

    Outputs()[0]->Reset(output_images);

    return true;
  }

 private:
  std::unique_ptr<GLStylizer> glStylizer_;
};

REGISTER_CPU_OPERATOR(OpenGLTextureToTensorStylizerDeprocess,
                      OpenGLTextureToTensorStylizerDeprocessOp);
OPERATOR_SCHEMA(OpenGLTextureToTensorStylizerDeprocess).NumInputs(2).NumOutputs(1);

REGISTER_CPU_OPERATOR(OpenGLTextureToTextureStylizerDeprocess,
                      OpenGLTextureToTextureStylizerDeprocessOp<RGBA>);
OPERATOR_SCHEMA(OpenGLTextureToTextureStylizerDeprocess).NumInputs(2).NumOutputs(1);
} // namespace caffe2
