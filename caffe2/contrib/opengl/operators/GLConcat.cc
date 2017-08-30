// Copyright 2004-present Facebook. All Rights Reserved.

#include "../core/GLFilter.h"
#include "../core/GLImage.h"
#include "../core/ImageAllocator.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include <iostream>
#include <vector>

class GLConcat : public GLFilter {
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

  GLConcat(int _batch_size = 1)
      : GLFilter("GLConcat",
                 vertex_shader,
                 fragment_shader,
                 input_bindings(_batch_size),
                 {/* no uniform blocks */},
                 {/* no attributes */},
                 {{"BATCH_SIZE", caffe2::to_string(_batch_size)}}),
        batch_size(_batch_size) {}

  template <typename T>
  void concat(const GLImageVector<T>** input_images,
              int size,
              const GLImageVector<T>& output_image);
  static const char* fragment_shader;
};

// MARK: GLSL

const char* GLConcat::fragment_shader = R"GLSL(#version 300 es

#define BATCH_SIZE $(BATCH_SIZE)

precision mediump float;
precision mediump int;

in highp vec2 v_texCoord;

uniform ivec2 outputSize;
//uniform ivec2 channelOffset;
TEXTURE_INPUT(inputData[BATCH_SIZE]);
//uniform sampler2D previousData[BATCH_SIZE];


TEXTURE_OUTPUT(0, outputData0);
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
  vec4 value = TEXTURE_LOAD(inputData[0], texelCoord);
  outputData0 = TEXTURE_STORE(value);
#if BATCH_SIZE > 1
  outputData1 = texelFetch(inputData[1], texelCoord, 0);
#if BATCH_SIZE > 2
  outputData2 = texelFetch(inputData[2], texelCoord, 0);
#if BATCH_SIZE > 3
  outputData3 = texelFetch(inputData[3], texelCoord, 0);
#endif
#endif
#endif
}

    )GLSL";

template <typename T>
void GLConcat::concat(const GLImageVector<T>** input_images,
                      int size,
                      const GLImageVector<T>& output_images) {
  for (int k = 0; k < output_images.size(); k++) {
    std::vector<int> input_slices;
    std::vector<int> input_slices_acc;
    input_slices_acc.push_back(0);
    for (int i = 0; i < size; i++) {
      int slices = (*input_images[i])[k]->slices;
      input_slices.push_back(slices);
      input_slices_acc.push_back(input_slices_acc[i] + slices);
    }
    GLImage<T>* output_image = output_images[k];
    int output_slices = output_image->slices;
    for (int is = 0; is < input_slices_acc[size]; is += batch_size) {
      std::vector<texture_attachment> input_attachments;
      for (int i = 0; i < batch_size; i++) {
        int image_index = 0;
        while (is + i >= input_slices_acc[image_index + 1]) {
          image_index += 1;
        }
        input_attachments.push_back(
            {(*input_images[image_index])[k]->textures[is + i - input_slices_acc[image_index]],
             inputData[i]});
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
class OpenGLConcatOp final : public Operator<CPUContext>, ImageAllocator<T> {
 public:
  OpenGLConcatOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        order_(StringToStorageOrder(OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    OPERATOR_NEEDS_FEATURE(this->order_ == StorageOrder::NCHW, "OpenGL only supports NCHW order.");
  }

  bool RunOnDevice() override {
    const GLImageVector<T>& input0 = Inputs()[0]->template Get<GLImageVector<T>>();
    const int num_images = input0.size();

    const GLImageVector<T>** input_images = new const GLImageVector<T>*[Inputs().size()];
    input_images[0] = &input0;
    int channelCount = input0.channels();
    // Only supports input channels divisible by 4 for now
    CAFFE_ENFORCE_EQ(input0.channels() % 4, 0);
    for (auto i = 1; i < Inputs().size(); i++) {
      const GLImageVector<T>& inputi = Inputs()[i]->template Get<GLImageVector<T>>();
      channelCount += inputi.channels();
      CAFFE_ENFORCE_EQ(num_images, inputi.size());
      CAFFE_ENFORCE_EQ(inputi.channels() % 4, 0);
      CAFFE_ENFORCE_EQ(input0.width(), inputi.width());
      CAFFE_ENFORCE_EQ(input0.height(), inputi.height());
      input_images[i] = &inputi;
    }

    const int input_width = input0.width();
    const int input_height = input0.height();

    const int output_channels = channelCount;
    const int output_width = input_width;
    const int output_height = input_height;

    int is_last = OperatorBase::GetSingleArgument<int>("is_last", 0);

    GLImageVector<T>* output = ImageAllocator<T>::newImage(
        num_images, output_width, output_height, output_channels, is_last);
    if (!_concat) {
      int batch_size = 1;

      batch_size = OperatorBase::GetSingleArgument<int>("batch_size", batch_size);
      _concat.reset(new GLConcat(batch_size));
    }

    _concat->concat(input_images, Inputs().size(), *output);
    delete[] input_images;
    Outputs()[0]->Reset(output);

    return true;
  }

 private:
  StorageOrder order_;
  std::unique_ptr<GLConcat> _concat;
};

REGISTER_CPU_OPERATOR(OpenGLConcat, OpenGLConcatOp<float16_t>);
OPERATOR_SCHEMA(OpenGLConcat).NumInputs(2, 4).NumOutputs(1, 2);
} // namespace caffe2
