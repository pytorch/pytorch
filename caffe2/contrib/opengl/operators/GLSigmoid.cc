// Copyright 2004-present Facebook. All Rights Reserved.

#include "../core/GLFilter.h"
#include "../core/GLImage.h"
#include "../core/ImageAllocator.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include <iostream>
#include <vector>

class GLSigmoid : public GLFilter {
 public:
  static constexpr int MaxBatchSize = 4;

  binding* inputData[MaxBatchSize];
  binding* outputSize;

  const int batch_size;

  const std::vector<binding*> input_bindings(int batch_size) {
    std::vector<binding*> bindings({BINDING(outputSize)});
    for (int i = 0; i < batch_size; i++) {
      bindings.push_back(inputData[i] = new binding{"inputData[" + caffe2::to_string(i) + "]"});
    }
    return bindings;
  }

  GLSigmoid(int _batch_size = 1)
      : GLFilter("GLSigmoid",
                 vertex_shader,
                 fragment_shader,
                 input_bindings(_batch_size),
                 {/* no uniform blocks */},
                 {/* no attributes */},
                 {{"BATCH_SIZE", caffe2::to_string(_batch_size)}}),
        batch_size(_batch_size) {}

  template <typename T>
  void sigmoid(const GLImageVector<T>& input_images, const GLImageVector<T>& output_images);

  static const char* fragment_shader;
};

// MARK: GLSL

const char* GLSigmoid::fragment_shader = R"GLSL(#version 300 es

#define BATCH_SIZE    $(BATCH_SIZE)

precision mediump float;
precision mediump int;
precision mediump sampler2D;

in highp vec2 v_texCoord;

uniform ivec2 outputSize;

uniform sampler2D inputData[BATCH_SIZE];

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

void main() {
    ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize));
    outputData0 = vec4(1.0) / (vec4(1.0) + exp(-texelFetch(inputData[0], ivec2(texelCoord), 0)));
    #if BATCH_SIZE > 1
    outputData1 = vec4(1.0) / (vec4(1.0) + exp(-texelFetch(inputData[1], ivec2(texelCoord), 0)));
    #if BATCH_SIZE > 2
    outputData2 = vec4(1.0) / (vec4(1.0) + exp(-texelFetch(inputData[2], ivec2(texelCoord), 0)));
    #if BATCH_SIZE > 3
    outputData3 = vec4(1.0) / (vec4(1.0) + exp(-texelFetch(inputData[3], ivec2(texelCoord), 0)));
#endif
#endif
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

    for (int is = 0; is < input_slices; is += batch_size) {

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
template <typename T>
class OpenGLSigmoidOp final : public Operator<CPUContext>, ImageAllocator<T> {
 public:
  OpenGLSigmoidOp(const OperatorDef& operator_def, Workspace* ws)
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

    if (!_sigmoid) {
      int batch_size = 1;
      batch_size = OperatorBase::GetSingleArgument<int>("batch_size", batch_size);
      _sigmoid.reset(new GLSigmoid(batch_size));
    }

    _sigmoid->sigmoid(input, *output);

    Outputs()[0]->Reset(output);

    return true;
  }

 private:
  StorageOrder order_;
  std::unique_ptr<GLSigmoid> _sigmoid;
};

REGISTER_CPU_OPERATOR(OpenGLSigmoid, OpenGLSigmoidOp<float16_t>);
OPERATOR_SCHEMA(OpenGLSigmoid)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape();
} // namespace caffe2
