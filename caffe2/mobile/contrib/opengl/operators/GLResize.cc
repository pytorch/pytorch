
#include "../core/GLFilter.h"
#include "../core/GLImage.h"
#include "../core/ImageAllocator.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include <iostream>
#include <vector>

class GLResizeNearest : public GLFilter {
 public:
  binding* inputData;
  binding* outputSize;
  binding* scale_reverse;

  GLResizeNearest()
      : GLFilter("GLResizeNearest",
                 vertex_shader,
                 fragment_shader,
                 std::vector<binding*>({BINDING(outputSize), BINDING(scale_reverse), BINDING(inputData)}),
                 {/* no uniform blocks*/},
                 {/* no attributes */},
                 {/* replacements */}) {}

  template <typename T>
  void resize(const GLImageVector<T>& input_images,
              const GLImageVector<T>& output_images,
              float width_scale_rev,
              float height_scale_rev);

  static const char* fragment_shader;
};

// MARK: GLSL

const char* GLResizeNearest::fragment_shader = R"GLSL(#version 300 es

precision mediump float;
precision mediump int;

in highp vec2 v_texCoord;

uniform ivec2 outputSize;
uniform highp vec2 scale_reverse;

TEXTURE_INPUT(inputData);
TEXTURE_OUTPUT(0, outputData);

void main() {
  // it clamps to the edge by default
  ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize) * scale_reverse);
  vec4 value = TEXTURE_LOAD(inputData, texelCoord);
  outputData = TEXTURE_STORE(value);
}
)GLSL";

template <typename T>
void GLResizeNearest::resize(const GLImageVector<T>& input_images,
                             const GLImageVector<T>& output_images,
                             float width_scale_rev,
                             float height_scale_rev) {
  for (int i = 0; i < input_images.size(); i++) {
    auto input_image = input_images[i];
    auto output_image = output_images[i];
    int input_slices = input_image->slices;
    int output_slices = output_image->slices;

    for (int is = 0; is < input_slices; is++) {
      std::vector<texture_attachment> input_attachments({{input_image->textures[is], inputData}});

      run(input_attachments,
          {output_image->textures.begin() + is, output_image->textures.begin() + is + 1},
          [&]() {
            glUniform2i(outputSize->location, output_image->texture_width, output_image->texture_height);
            glUniform2f(scale_reverse->location, width_scale_rev, height_scale_rev);
          },
          output_image->texture_width,
          output_image->texture_height);
    }
  }
}

namespace caffe2 {

template <class T>
class OpenGLResizeNearestOp final : public Operator<CPUContext>, ImageAllocator<T> {
 public:
  OpenGLResizeNearestOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws), width_scale_(1), height_scale_(1) {
    if (HasArgument("width_scale")) {
      width_scale_ = static_cast<float>(OperatorBase::GetSingleArgument<float>("width_scale", 1));
    }
    if (HasArgument("height_scale")) {
      height_scale_ = static_cast<float>(OperatorBase::GetSingleArgument<float>("height_scale", 1));
    }
  }

  bool RunOnDevice() override {
    const GLImageVector<T>& input = Inputs()[0]->template Get<GLImageVector<T>>();
    const int num_images = input.size();
    const int input_width = input.width();
    const int input_height = input.height();
    const int input_channels = input.channels();

    const int output_width = input_width * width_scale_;
    const int output_height = input_height * height_scale_;
    const int output_channels = input_channels;

    const int input_tile_x = input.tile_x(), input_tile_y = input.tile_y();
    const int output_tile_x = input_tile_x, output_tile_y = input_tile_y;

    int is_last = OperatorBase::GetSingleArgument<int>("is_last", 0);
    GLImageVector<T>* output = ImageAllocator<T>::newImage(
        num_images, output_width, output_height, output_channels, output_tile_x, output_tile_y, is_last);

    if (!resizeNearest_) {
      resizeNearest_.reset(new GLResizeNearest());
    }
    resizeNearest_->resize(input, *output, 1.0 / width_scale_, 1.0 / height_scale_);
    Outputs()[0]->Reset(output);

    return true;
  }

 protected:
  float width_scale_;
  float height_scale_;
  std::unique_ptr<GLResizeNearest> resizeNearest_;
};

REGISTER_CPU_OPERATOR(OpenGLResizeNearest, OpenGLResizeNearestOp<float16_t>);
OPERATOR_SCHEMA(OpenGLResizeNearest).NumInputs(1).NumOutputs(1);

} // namespace caffe2
