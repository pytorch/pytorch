
#include "../core/GLFilter.h"
#include "../core/GLImage.h"
#include "../core/ImageAllocator.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include <iostream>
#include <vector>

class GLAdd : public GLFilter {
 public:
  binding* inputData[2];
  binding* outputSize;

  GLAdd()
      : GLFilter("GLAdd",
                 vertex_shader,
                 fragment_shader,
                 std::vector<binding*>(
                     {BINDING(outputSize), BINDING(inputData[0]), BINDING(inputData[1])}),
                 {/* no uniform blocks */},
                 {/* no attributes */},
                 {/* no replacements */}) {}

  template <typename T>
  void add(const GLImageVector<T>& input_image0,
           const GLImageVector<T>& input_image1,
           const GLImageVector<T>& output_image);

  static const char* fragment_shader;
};

// MARK: GLSL

const char* GLAdd::fragment_shader = R"GLSL(#version 300 es

precision mediump float;
precision mediump int;

in highp vec2 v_texCoord;

uniform ivec2 outputSize;

TEXTURE_INPUT(inputData[2]);
TEXTURE_OUTPUT(0, outputData);

void main() {
    ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize));
    vec4 A = TEXTURE_LOAD(inputData[0], texelCoord);
    vec4 B = TEXTURE_LOAD(inputData[1], texelCoord);
    vec4 value = A + B;
    outputData = TEXTURE_STORE(value);
}

)GLSL";

template <typename T>
void GLAdd::add(const GLImageVector<T>& input_images0,
                const GLImageVector<T>& input_images1,
                const GLImageVector<T>& output_images) {
  const int num_images = input_images0.size();
  for (int i = 0; i < num_images; i++) {
    GLImage<T>* input_image0 = input_images0[i];
    GLImage<T>* input_image1 = input_images1[i];
    int input_slices = input_image0->slices;
    GLImage<T>* output_image = output_images[i];
    int output_slices = output_image->slices;

    for (int is = 0; is < input_slices; is++) {
      std::vector<texture_attachment> input_attachments;
      input_attachments.push_back({input_image0->textures[is], inputData[0]});
      input_attachments.push_back({input_image1->textures[is], inputData[1]});

      run(input_attachments,
          {output_image->textures.begin() + is, output_image->textures.begin() + is + 1},
          [&]() { glUniform2i(outputSize->location, output_image->texture_width, output_image->texture_height); },
          output_image->texture_width,
          output_image->texture_height);
    }
  }
}

namespace caffe2 {
template <typename T>
class OpenGLAddOp final : public Operator<CPUContext>, ImageAllocator<T> {
 public:
  OpenGLAddOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(OperatorBase::HasArgument("broadcast") == false,
                           "OpenGLAdd does not support broadcast");

    OPERATOR_NEEDS_FEATURE(OperatorBase::HasArgument("axis") == false, "OpenGLAdd does not support axis");
  }

  bool RunOnDevice() override {
    const GLImageVector<T>& input0 = Inputs()[0]->template Get<GLImageVector<T>>();
    const GLImageVector<T>& input1 = Inputs()[1]->template Get<GLImageVector<T>>();

    CAFFE_ENFORCE_EQ(input0.size(), input1.size());

    const int num_images = input0.size();
    const int input_channels = input0.channels();
    const int input_width = input0.width();
    const int input_height = input0.height();
    const int input_tile_x   = input0.tile_x();
    const int input_tile_y   = input0.tile_y();

    CAFFE_ENFORCE_EQ(input1.channels(), input_channels);
    CAFFE_ENFORCE_EQ(input1.width(), input_width);
    CAFFE_ENFORCE_EQ(input1.height(), input_height);
    CAFFE_ENFORCE_EQ(input1.tile_x(), input_tile_x);
    CAFFE_ENFORCE_EQ(input1.tile_y(), input_tile_y);

    const int output_channels = input_channels;
    const int output_width = input_width;
    const int output_height = input_height;
    const int output_tile_x   = input_tile_x;
    const int output_tile_y   = input_tile_y;

    int is_last = OperatorBase::GetSingleArgument<int>("is_last", 0);

    GLImageVector<T>* output = ImageAllocator<T>::newImage(
        num_images, output_width, output_height, output_channels, output_tile_x, output_tile_y, is_last);

    if (!_add) {
      _add.reset(new GLAdd());
    }

    _add->add(input0, input1, *output);

    Outputs()[0]->Reset(output);

    return true;
  }

 private:
  std::unique_ptr<GLAdd> _add;
};

REGISTER_CPU_OPERATOR(OpenGLAdd, OpenGLAddOp<float16_t>);
OPERATOR_SCHEMA(OpenGLAdd).NumInputs(2).NumOutputs(1);
} // namespace caffe2
