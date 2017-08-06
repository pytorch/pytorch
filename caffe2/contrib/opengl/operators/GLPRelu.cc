// Copyright 2004-present Facebook. All Rights Reserved.

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

  static constexpr int MaxBatchSize = 4;

  const float* bias;

  binding* inputData[MaxBatchSize];
  binding* outputSize;
  binding* bias_block;

  const int bias_size;
  const int channels;
  const int batch_size;

  const std::vector<binding*> input_bindings(int batch_size) {
    std::vector<binding*> bindings({BINDING(outputSize)});
    for (int i = 0; i < batch_size; i++) {
      bindings.push_back(inputData[i] = new binding{"inputData[" + caffe2::to_string(i) + "]"});
    }
    return bindings;
  }

  GLPRelu(const float* _bias, const int _bias_size, const int _channels, int _batch_size = 1)
      : GLFilter("GLPRelu",
                 vertex_shader,
                 fragment_shader,
                 input_bindings(_batch_size),
                 std::vector<binding*>({BINDING(bias_block)}),
                 {/* no attributes */},
                 {{"BATCH_SIZE", caffe2::to_string(_batch_size)},
                  {"USE_RELU", caffe2::to_string(PRelu)}}),
        bias(_bias),
        bias_size(_bias_size),
        channels(_channels),
        batch_size(_batch_size) {
    attach_uniform_buffer(bias_block, 0, nullptr);
  }

  GLPRelu(const int _channels, int _batch_size = 1)
      : GLFilter("GLRelu",
                 vertex_shader,
                 fragment_shader,
                 input_bindings(_batch_size),
                 {/* no uniform blocks */},
                 {/* no attributes */},
                 {{"BATCH_SIZE", caffe2::to_string(_batch_size)},
                  {"USE_RELU", caffe2::to_string(Relu)}}),
        bias(nullptr),
        bias_block(nullptr),
        bias_size(0),
        channels(_channels),
        batch_size(_batch_size) {}

  template <typename T>
  void prelu(const GLImageVector<T>& input_images,
             const GLImageVector<T>& output_images,
             GLPRelu::ReluType reluType);

  static const char* fragment_shader;
};

// MARK: GLSL

const char* GLPRelu::fragment_shader = R"GLSL(#version 300 es

#define BATCH_SIZE    $(BATCH_SIZE)
#define USE_RELU      $(USE_RELU)

precision mediump float;
precision mediump int;
precision mediump sampler2D;

in highp vec2 v_texCoord;

uniform ivec2 outputSize;

uniform sampler2D inputData[BATCH_SIZE];

#if !USE_RELU
layout (std140) uniform bias_block {
  highp uvec4 bias[(BATCH_SIZE+1)/2];
};

#define unpackHalf4x16(pd) vec4(unpackHalf2x16(pd.x), unpackHalf2x16(pd.y))
#endif

layout(location = 0) out mediump vec4 outputData0;
#if BATCH_SIZE > 1
layout(location = 1) out mediump vec4 outputData1;
#if BATCH_SIZE > 2
layout(location = 2) out mediump vec4 outputData2;
#if BATCH_SIZE > 3
layout(location = 3) out mediump vec4 outputData3;
#endif
#endif
#endif

#if !USE_RELU
void main() {
  ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize));

  // output.data     = value > 0 ? value : value * weight;
  vec4 value = texelFetch(inputData[0], texelCoord, 0);
  outputData0 = mix(value * unpackHalf4x16(bias[0].xy), value, vec4(greaterThan(value, vec4(0))));
#if BATCH_SIZE > 1
  value = texelFetch(inputData[1], texelCoord, 0);
  outputData1 = mix(value * unpackHalf4x16(bias[0].zw), value, vec4(greaterThan(value, vec4(0))));
#if BATCH_SIZE > 2
  value = texelFetch(inputData[2], texelCoord, 0);
  outputData2 = mix(value * unpackHalf4x16(bias[1].xy), value, vec4(greaterThan(value, vec4(0))));
#if BATCH_SIZE > 3
  value = texelFetch(inputData[3], texelCoord, 0);
  outputData3 = mix(value * unpackHalf4x16(bias[1].zw), value, vec4(greaterThan(value, vec4(0))));
#endif
#endif
#endif
}

#else
void main() {
    ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize));

    outputData0 = max(texelFetch(inputData[0], texelCoord, 0), vec4(0.0));
#if BATCH_SIZE > 1
    outputData1 = max(texelFetch(inputData[1], texelCoord, 0), vec4(0.0));
#if BATCH_SIZE > 2
    outputData2 = max(texelFetch(inputData[2], texelCoord, 0), vec4(0.0));
#if BATCH_SIZE > 3
    outputData3 = max(texelFetch(inputData[3], texelCoord, 0), vec4(0.0));
#endif
#endif
#endif
}
#endif

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

    gl_log(GL_VERBOSE, "batch_size: %d\n", batch_size);
    for (int is = 0; is < input_slices; is += batch_size) {
      gl_log(GL_VERBOSE, "is: %d\n", is);

      if (reluType == PRelu) {
        attach_uniform_buffer(bias_block, 0, [&](float16_t* data, size_t size) {
          if (size != (4 * ((batch_size + 1) / 2 * 2) * sizeof(float16_t))) {
            std::cerr << "size: " << size << ", (4 * ((BATCH_SIZE+1)/2 * 2) * sizeof(float16_t)): "
                      << (4 * ((batch_size + 1) / 2 * 2) * sizeof(float16_t)) << "\n";
            throw std::runtime_error("Bias size mismatch");
          }

          for (int i = 0; i < std::min(4 * batch_size, channels - 4 * is); i++) {
            data[i] = bias_size == channels ? bias[4 * is + i] : bias[0];
          }
        });
      }

      std::vector<texture_attachment> input_attachments;
      for (int ib = 0; ib < batch_size; ib++) {
        input_attachments.push_back({input_image->textures[is + ib], inputData[ib]});
      }

      run(input_attachments,
          {output_image->textures.begin() + is, output_image->textures.begin() + is + batch_size},
          [&]() { glUniform2i(outputSize->location, output_image->width, output_image->height); },
          output_image->width,
          output_image->height);
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

    GLImageVector<T>* output = ImageAllocator<T>::newImage(
        num_images, output_width, output_height, output_channels, is_last);

    const auto* scale = reluType == GLPRelu::PRelu ? &Input(1) : nullptr;

    if (!_prelu) {
      int batch_size = 1;
      batch_size = OperatorBase::GetSingleArgument<int>("batch_size", batch_size);
      // LOG(INFO) << "batch_size = " << batch_size;
      if (reluType == GLPRelu::PRelu) {
        _prelu.reset(
            new GLPRelu(scale->template data<float>(), scale->size(), input_channels, batch_size));
      } else {
        _prelu.reset(new GLPRelu(input_channels, batch_size));
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
