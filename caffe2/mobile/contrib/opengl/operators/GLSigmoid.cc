
#include "../core/GLFilter.h"
#include "../core/GLImage.h"
#include "../core/ImageAllocator.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include <iostream>
#include <vector>

typedef enum { Sigmoid, Tanh } OpType;

class GLSigmoid : public GLFilter {
 public:
  binding* inputData;
  binding* outputSize;

  GLSigmoid(OpType opType)
      : GLFilter("GLSigmoid",
                 vertex_shader,
                 fragment_shader,
                 {BINDING(outputSize), BINDING(inputData)},
                 {/* no uniform blocks */},
                 {/* no attributes */},
                 {{"SIGMOID", caffe2::to_string(opType == Sigmoid)},
                  {"TANH", caffe2::to_string(opType == Tanh)}}) {}

  template <typename T>
  void sigmoid(const GLImageVector<T>& input_images, const GLImageVector<T>& output_images);

  static const char* fragment_shader;
};

// MARK: GLSL

const char* GLSigmoid::fragment_shader = R"GLSL(#version 300 es
#define SIGMOID $(SIGMOID)
#define TANH $(TANH)

precision mediump float;
precision mediump int;

in highp vec2 v_texCoord;

uniform ivec2 outputSize;

TEXTURE_INPUT(inputData);
TEXTURE_OUTPUT(0, outputData);

void main() {
  ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize));
  vec4 value = TEXTURE_LOAD(inputData, ivec2(texelCoord));
#if SIGMOID
  value = vec4(1.0) / (vec4(1.0) + exp(-value));
  outputData = TEXTURE_STORE(value);
#elif TANH
  value = tanh(value);
  outputData = TEXTURE_STORE(value);
#endif
}

)GLSL";

template <typename T>
void GLSigmoid::sigmoid(const GLImageVector<T>& input_images,
                        const GLImageVector<T>& output_images) {
  for (int i = 0; i < input_images.size(); i++) {
    auto input_image = input_images[i];
    auto output_image = output_images[i];
    int input_slices = input_image->slices;
    int output_slices = output_image->slices;

    for (int is = 0; is < input_slices; is++) {
      run(std::vector<texture_attachment>({{input_image->textures[is], inputData}}),
          {output_image->textures.begin() + is, output_image->textures.begin() + is + 1},
          [&]() { glUniform2i(outputSize->location, output_image->width, output_image->height); },
          output_image->width,
          output_image->height);
    }
  }
}

namespace caffe2 {
template <typename T, OpType opType>
class OpenGLSigmoidOp final : public Operator<CPUContext>, ImageAllocator<T> {
 public:
  OpenGLSigmoidOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

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

    if (!_sigmoid) {
      _sigmoid.reset(new GLSigmoid(opType));
    }

    _sigmoid->sigmoid(input, *output);

    Outputs()[0]->Reset(output);

    return true;
  }

 private:
  std::unique_ptr<GLSigmoid> _sigmoid;
};

REGISTER_CPU_OPERATOR(OpenGLSigmoid, OpenGLSigmoidOp<float16_t, Sigmoid>);
OPERATOR_SCHEMA(OpenGLSigmoid)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape();

REGISTER_CPU_OPERATOR(OpenGLTanh, OpenGLSigmoidOp<float16_t, Tanh>);
OPERATOR_SCHEMA(OpenGLTanh)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape();
} // namespace caffe2
