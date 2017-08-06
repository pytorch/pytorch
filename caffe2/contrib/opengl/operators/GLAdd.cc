// Copyright 2004-present Facebook. All Rights Reserved.

#include "../core/GLFilter.h"
#include "../core/GLImage.h"
#include "../core/ImageAllocator.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include <iostream>
#include <vector>

class GLAdd : public GLFilter {
 public:
  static constexpr int MaxBatchSize = 4;

  binding* inputData[2 * MaxBatchSize];
  binding* outputSize;

  const int batch_size;

  const std::vector<binding*> input_bindings(int batch_size) {
    std::vector<binding*> bindings({BINDING(outputSize)});
    for (int i = 0; i < 2 * batch_size; i++) {
      bindings.push_back(inputData[i] = new binding{"inputData[" + caffe2::to_string(i) + "]"});
    }
    return bindings;
  }

  GLAdd(int _batch_size = 1)
      : GLFilter("GLAdd",
                 vertex_shader,
                 fragment_shader,
                 input_bindings(_batch_size),
                 {/* no uniform blocks */},
                 {/* no attributes */},
                 {{"BATCH_SIZE", caffe2::to_string(_batch_size)}}),
        batch_size(_batch_size) {}

  template <typename T>
  void add(const GLImageVector<T>& input_image0,
           const GLImageVector<T>& input_image1,
           const GLImageVector<T>& output_image);

  static const char* fragment_shader;
};

// MARK: GLSL

const char* GLAdd::fragment_shader = R"GLSL(#version 300 es

#define BATCH_SIZE    $(BATCH_SIZE)

precision mediump float;
precision mediump int;
precision mediump sampler2D;

in highp vec2 v_texCoord;

uniform ivec2 outputSize;

uniform sampler2D inputData[2 * BATCH_SIZE];

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
    outputData0 = texelFetch(inputData[0], texelCoord, 0) + texelFetch(inputData[1], texelCoord, 0);
#if BATCH_SIZE > 1
    outputData1 = texelFetch(inputData[2], texelCoord, 0) + texelFetch(inputData[3], texelCoord, 0);
#if BATCH_SIZE > 2
    outputData2 = texelFetch(inputData[4], texelCoord, 0) + texelFetch(inputData[5], texelCoord, 0);
#if BATCH_SIZE > 3
    outputData3 = texelFetch(inputData[6], texelCoord, 0) + texelFetch(inputData[7], texelCoord, 0);
#endif
#endif
#endif
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
    int input_slices0 = input_image0->slices;
    int input_slices1 = input_image1->slices;
    GLImage<T>* output_image = output_images[i];
    int output_slices = output_image->slices;
    gl_log(GL_VERBOSE, "batch_size: %d\n", batch_size);
    for (int is = 0; is < input_slices0; is += batch_size) {
      gl_log(GL_VERBOSE, "is: %d\n", is);

      std::vector<texture_attachment> input_attachments;
      for (int i = 0; i < batch_size; i++) {
        input_attachments.push_back({input_image0->textures[is + i], inputData[2 * i]});
        input_attachments.push_back({input_image1->textures[is + i], inputData[2 * i + 1]});
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
class OpenGLAddOp final : public Operator<CPUContext>, ImageAllocator<T> {
 public:
  OpenGLAddOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        order_(StringToStorageOrder(OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW, "OpenGL only supports NCHW order.");
  }

  bool RunOnDevice() override {
    const GLImageVector<T>& input0 = Inputs()[0]->template Get<GLImageVector<T>>();
    const GLImageVector<T>& input1 = Inputs()[1]->template Get<GLImageVector<T>>();

    CAFFE_ENFORCE_EQ(input0.size(), input1.size());

    const int num_images = input0.size();
    const int input_channels = input0.channels();
    const int input_width = input0.width();
    const int input_height = input0.height();
    CAFFE_ENFORCE_EQ(input1.channels(), input_channels);
    CAFFE_ENFORCE_EQ(input1.width(), input_width);
    CAFFE_ENFORCE_EQ(input1.height(), input_height);

    const int output_channels = input_channels;
    const int output_width = input_width;
    const int output_height = input_height;

    int is_last = OperatorBase::GetSingleArgument<int>("is_last", 0);

    GLImageVector<T>* output = ImageAllocator<T>::newImage(
        num_images, output_width, output_height, output_channels, is_last);

    if (!_add) {
      int batch_size = 1;

      batch_size = OperatorBase::GetSingleArgument<int>("batch_size", batch_size);
      _add.reset(new GLAdd(batch_size));
    }

    _add->add(input0, input1, *output);

    Outputs()[0]->Reset(output);

    return true;
  }

 private:
  StorageOrder order_;
  std::unique_ptr<GLAdd> _add;
};

REGISTER_CPU_OPERATOR(OpenGLAdd, OpenGLAddOp<float16_t>);
OPERATOR_SCHEMA(OpenGLAdd).NumInputs(2).NumOutputs(1);
} // namespace caffe2
