
#include "../core/GLFilter.h"
#include "../core/GLImage.h"
#include "../core/ImageAllocator.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"

class GLMul : public GLFilter {
 public:
  binding* outputSize;
  binding* inputData;
  binding* B;

  GLMul()
      : GLFilter("GLMul",
                 vertex_shader,
                 fragment_shader,
                 std::vector<binding*>({BINDING(outputSize), BINDING(inputData), BINDING(B)}),
                 {/* no uniform blocks */},
                 {/* no attributes */},
                 {/* no replacements */}) {}

  template <typename T>
  void mul(const GLImageVector<T>& input_images, const GLImageVector<T>& output_images, float b);

  static const char* fragment_shader;
};

// MARK: GLSL

const char* GLMul::fragment_shader = R"GLSL(#version 300 es

precision mediump float;
precision mediump int;

in highp vec2 v_texCoord;

uniform ivec2 outputSize;
uniform vec4 B;

TEXTURE_INPUT(inputData);
TEXTURE_OUTPUT(0, outputData);

void main() {
  ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize));
  vec4 A = TEXTURE_LOAD(inputData, texelCoord);
  outputData = TEXTURE_STORE(A * B);
}

)GLSL";

template <typename T>
void GLMul::mul(const GLImageVector<T>& input_images,
                const GLImageVector<T>& output_images,
                float b) {
  for (int i = 0; i < input_images.size(); i++) {
    auto input_image = input_images[i];
    auto output_image = output_images[i];
    int input_slices = input_image->slices;
    int output_slices = output_image->slices;

    for (int is = 0; is < input_slices; is++) {
      run(std::vector<texture_attachment>({{input_image->textures[is], inputData}}),
          {output_image->textures.begin() + is, output_image->textures.begin() + is + 1},
          [&]() {
            glUniform2i(outputSize->location, output_image->width, output_image->height);
            glUniform4f(B->location, b, b, b, b);
          },
          output_image->width,
          output_image->height);
    }
  }
}

namespace caffe2 {
template <class T>
class OpenGLMulOp final : public Operator<CPUContext>, ImageAllocator<T> {
 public:
  OpenGLMulOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(OperatorBase::GetSingleArgument<int>("broadcast", 0) == 1,
                           "OpenGLMul only supports broadcast");

    OPERATOR_NEEDS_FEATURE(OperatorBase::HasArgument("axis") == false,
                           "OpenGLMul does not support axis");
  }

  bool RunOnDevice() override {
    const GLImageVector<T>& input = Inputs()[0]->template Get<GLImageVector<T>>();
    const auto& B = Input(1);
    CAFFE_ENFORCE_EQ(B.size(), 1); // only scalar is supported

    const int num_images = input.size();
    const auto output_height = input.height();
    const auto output_width = input.width();
    const int output_channels = input.channels();

    int is_last = OperatorBase::GetSingleArgument<int>("is_last", 0);

    GLImageVector<T>* output = ImageAllocator<T>::newImage(
        num_images, output_width, output_height, output_channels, is_last);

    if (!_mult) {
      _mult.reset(new GLMul());
    }

    _mult->mul(input, *output, B.template data<float>()[0]);

    Outputs()[0]->Reset(output);

    return true;
  }

 private:
  std::unique_ptr<GLMul> _mult;
};

REGISTER_CPU_OPERATOR(OpenGLMul, OpenGLMulOp<float16_t>);
} // namespace caffe2
