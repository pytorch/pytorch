// Copyright 2004-present Facebook. All Rights Reserved.

#include "../core/GLFilter.h"
#include "../core/GLImage.h"
#include "../core/ImageAllocator.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include <iostream>
#include <vector>

class GLResizeNearest : public GLFilter {
 public:
  static constexpr int MaxBatchSize = 4;

  binding* inputData[MaxBatchSize];
  binding* inputSize;
  binding* outputSize;
  binding* scale_reverse;

  const int batch_size;

  const std::vector<binding*> input_bindings(int batch_size) {
    std::vector<binding*> bindings(
        {BINDING(inputSize), BINDING(outputSize), BINDING(scale_reverse)});
    for (int i = 0; i < batch_size; i++) {
      bindings.push_back(inputData[i] = new binding{"inputData[" + caffe2::to_string(i) + "]"});
    }
    return bindings;
  }

  GLResizeNearest(int _batch_size = 1)
      : GLFilter("GLResizeNearest",
                 vertex_shader,
                 fragment_shader,
                 input_bindings(_batch_size),
                 {/* no uniform blocks*/},
                 {/* no attributes */},
                 {{"BATCH_SIZE", caffe2::to_string(_batch_size)}}),
        batch_size(_batch_size) {}

  template <typename T>
  void resize(const GLImageVector<T>& input_images,
              const GLImageVector<T>& output_images,
              float width_scale_rev,
              float height_scale_rev);

  static const char* fragment_shader;
};

// MARK: GLSL

const char* GLResizeNearest::fragment_shader = R"GLSL(#version 300 es

#define BATCH_SIZE    $(BATCH_SIZE)

precision mediump float;
precision mediump int;

in highp vec2 v_texCoord;

uniform ivec2 inputSize;
uniform ivec2 outputSize;
uniform highp vec2 scale_reverse;

TEXTURE_INPUT(inputData[BATCH_SIZE]);

TEXTURE_OUTPUT(0, outputData0);
#if BATCH_SIZE > 1
TEXTURE_OUTPUT(1, outputData1);
#if BATCH_SIZE > 2
TEXTURE_OUTPUT(2, outputData2);
#if BATCH_SIZE > 3
TEXTURE_OUTPUT(3, outputData3);
#endif
#endif
#endif

void main() {
  // it clamps to the edge by default
  ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize) * scale_reverse);

  vec4 v0 = TEXTURE_LOAD(inputData[0], texelCoord);
  outputData0 = TEXTURE_STORE(v0);
#if BATCH_SIZE > 1
  vec4 v1 = TEXTURE_LOAD(inputData[1], texelCoord);
  outputData1 = TEXTURE_STORE(v1);
#if BATCH_SIZE > 2
  vec4 v2 = TEXTURE_LOAD(inputData[2], texelCoord);
  outputData2 = TEXTURE_STORE(v2);
#if BATCH_SIZE > 3
  vec4 v3 = TEXTURE_LOAD(inputData[3], texelCoord);
  outputData3 = TEXTURE_STORE(v3);
#endif
#endif
#endif
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

    for (int is = 0; is < input_slices; is += batch_size) {
      std::vector<texture_attachment> input_attachments;
      for (int ib = 0; ib < batch_size; ib++) {
        input_attachments.push_back({input_image->textures[is + ib], inputData[ib]});
      }

      run(input_attachments,
          {output_image->textures.begin() + is, output_image->textures.begin() + is + batch_size},
          [&]() {
            glUniform2i(inputSize->location, input_image->width, input_image->height);
            glUniform2i(outputSize->location, output_image->width, output_image->height);
            glUniform2f(scale_reverse->location, width_scale_rev, height_scale_rev);
          },
          output_image->width,
          output_image->height);
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

    int is_last = OperatorBase::GetSingleArgument<int>("is_last", 0);
    GLImageVector<T>* output = ImageAllocator<T>::newImage(
        num_images, output_width, output_height, output_channels, is_last);

    if (!resizeNearest_) {
      int batch_size = OperatorBase::GetSingleArgument<int>("batch_size", 1);
      resizeNearest_.reset(new GLResizeNearest(batch_size));
      LOG(INFO) << "batch_size = " << batch_size;
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
