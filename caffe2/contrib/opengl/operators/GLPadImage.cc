// Copyright 2004-present Facebook. All Rights Reserved.

#include "../core/GLFilter.h"
#include "../core/GLImage.h"
#include "../core/ImageAllocator.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "caffe2/operators/conv_pool_op_base.h"

class GLPadImage : public GLFilter {
 public:
  binding* padSize;
  binding* inputSize;
  binding* outputSize;
  binding* inputData;

  GLPadImage()
      : GLFilter(
            "GLPadImage",
            vertex_shader,
            fragment_shader,
            std::vector<binding*>(
                {BINDING(padSize), BINDING(inputSize), BINDING(outputSize), BINDING(inputData)}),
            {/* no uniform blocks */},
            {/* no attributes */},
            {/* no replacements */}) {}

  template <typename T>
  void pad(const GLImageVector<T>& input_images,
           const GLImageVector<T>& output_images,
           const int pad_l,
           const int pad_t);

  static const char* fragment_shader;
};

// MARK: GLSL

const char* GLPadImage::fragment_shader = R"GLSL(#version 300 es

precision mediump float;
precision mediump int;
precision mediump sampler2D;

in highp vec2 v_texCoord;

uniform ivec2 padSize;
uniform ivec2 inputSize;
uniform ivec2 outputSize;

uniform sampler2D inputData;
layout(location = 0) out mediump vec4 outputData;

void main() {
  ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize)) - padSize;
  texelCoord = max(texelCoord, -texelCoord);
  texelCoord = min(texelCoord, ivec2(2) * (inputSize - 1) - texelCoord);
  outputData = texelFetch(inputData, texelCoord, 0);
}

)GLSL";

template <typename T>
void GLPadImage::pad(const GLImageVector<T>& input_images,
                     const GLImageVector<T>& output_images,
                     const int pad_l,
                     const int pad_t) {
  for (int i = 0; i < input_images.size(); i++) {
    auto input_image = input_images[i];
    auto output_image = output_images[i];
    int input_slices = input_image->slices;
    int output_slices = output_image->slices;

    for (int is = 0; is < input_slices; is++) {
      gl_log(GL_VERBOSE, "is: %d\n", is);
      run(std::vector<texture_attachment>({{input_image->textures[is], inputData}}),
          {output_image->textures.begin() + is, output_image->textures.begin() + is + 1},
          [&]() {
            glUniform2i(inputSize->location, input_image->width, input_image->height);
            glUniform2i(outputSize->location, output_image->width, output_image->height);
            glUniform2i(padSize->location, pad_l, pad_t);
          },
          output_image->width,
          output_image->height);
    }
  }
}

namespace caffe2 {
template <class T>
class OpenGLPadImageOp final : public ConvPoolOpBase<CPUContext>, ImageAllocator<T> {
 public:
  OpenGLPadImageOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(order_ == StorageOrder::NCHW, "OpenGL only supports NCHW order.");

    OPERATOR_NEEDS_FEATURE(OperatorBase::GetSingleArgument<string>("mode", "") == "reflect",
                           "OpenGL only supports reflection");
    CAFFE_ENFORCE(legacy_pad_ == LegacyPadding::NOTSET,
                  "Padding layer only supports explicit pad values.");
    CAFFE_ENFORCE(dilation_h() == 1 && dilation_w() == 1,
                  "Pooling op does not support dilation right now.");
    CAFFE_ENFORCE(stride_h() == 1 && stride_w() == 1,
                  "Pooling op does not support stride right now.");
    // Pad op does not use kernel sizes, so we set it to 1 for computing the
    // output size.
    kernel_[0] = kernel_[1] = 1;
  }

  bool RunOnDeviceWithOrderNCHW() override {
    const GLImageVector<T>& input = Inputs()[0]->template Get<GLImageVector<T>>();
    const auto pH = pad_t();
    const auto pW = pad_l();
    const int num_images = input.size();
    const auto output_height = input.height() + 2 * pH;
    const auto output_width = input.width() + 2 * pW;
    const int output_channels = input.channels();

    int is_last = OperatorBase::GetSingleArgument<int>("is_last", 0);

    GLImageVector<T>* output = ImageAllocator<T>::newImage(
        num_images, output_width, output_height, output_channels, is_last);

    if (!_padImage) {
      _padImage.reset(new GLPadImage());
    }

    _padImage->pad(input, *output, pW, pH);

    Outputs()[0]->Reset(output);

    return true;
  }

 private:
  std::unique_ptr<GLPadImage> _padImage;
};

REGISTER_CPU_OPERATOR(OpenGLPadImage, OpenGLPadImageOp<float16_t>);
OPERATOR_SCHEMA(OpenGLPadImage).NumInputs(1).NumOutputs(1);
} // namespace caffe2
