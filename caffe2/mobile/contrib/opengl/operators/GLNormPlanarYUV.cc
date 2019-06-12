
#include "../core/GLFilter.h"
#include "../core/GLImage.h"
#include "../core/ImageAllocator.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include <iostream>
#include <vector>

class GLNormPlanarYUV : public GLFilter {
 public:
  const float* mean;
  const float* std;

  binding* inputData;
  binding* outputSize;
  binding* mean_data;
  binding* std_data;

  GLNormPlanarYUV(const float* _mean, const float* _std)
      : GLFilter("GLNormPlanarYUV",
                 vertex_shader,
                 fragment_shader,
                 std::vector<binding*>({BINDING(inputData),
                                        BINDING(outputSize),
                                        BINDING(mean_data),
                                        BINDING(std_data)}), // input bindings
                 {/* no uniform blocks */},
                 {/* no attributes */},
                 {}),
        mean(_mean),
        std(_std) {}

  template <typename T>
  void normalize(const GLImageVector<T>& input_images, const GLImageVector<T>& output_images);

  static const char* fragment_shader;
};

// MARK: GLSL

const char* GLNormPlanarYUV::fragment_shader = R"GLSL(#version 300 es

precision mediump float;
precision mediump int;

in highp vec2 v_texCoord;

uniform ivec2 outputSize;
uniform vec4 mean_data;
uniform vec4 std_data;

TEXTURE_INPUT(inputData);
TEXTURE_OUTPUT(0, outputData);

void main() {
  ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize));
  vec4 value = TEXTURE_LOAD(inputData, texelCoord);
  outputData = TEXTURE_STORE((value - mean_data) / std_data);
}

)GLSL";

template <class T>
void GLNormPlanarYUV::normalize(const GLImageVector<T>& input_images,
                                const GLImageVector<T>& output_images) {
  int num_images = input_images.size();
  for (int i = 0; i < num_images; i++) {
    GLImage<T>* input_image = input_images[i];
    GLImage<T>* output_image = output_images[i];
    int input_slices = input_image->slices;
    int output_slices = output_image->slices;

    for (int is = 0; is < input_slices; is++) {

      std::vector<texture_attachment> input_attachments({{input_image->textures[is], inputData}});

      run(input_attachments,
          {output_image->textures.begin() + is, output_image->textures.begin() + is + 1},
          [&]() {
            glUniform2i(outputSize->location, output_image->width, output_image->height);
            glUniform4f(mean_data->location, mean[0], mean[1], mean[2], 0.0);
            glUniform4f(std_data->location, std[0], std[1], std[2], 1.0);
          },
          output_image->width,
          output_image->height);
    }
  }
}

namespace caffe2 {
template <typename T>
class GLNormPlanarYUVOp final : public Operator<CPUContext>, ImageAllocator<T> {
 public:
  GLNormPlanarYUVOp(const OperatorDef& operator_def, Workspace* ws)
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

    const auto& M = Input(1); // mean
    const auto& S = Input(2); // standard deviation
    CAFFE_ENFORCE(input_channels == M.dim(1));
    CAFFE_ENFORCE(input_channels == S.dim(1));

    if (!_normPlanarYUV) {
      _normPlanarYUV.reset(new GLNormPlanarYUV(M.template data<float>(), S.template data<float>()));
    }

    _normPlanarYUV->normalize(input, *output);

    Outputs()[0]->Reset(output);

    return true;
  }

 private:
  StorageOrder order_;
  std::unique_ptr<GLNormPlanarYUV> _normPlanarYUV;
};

REGISTER_CPU_OPERATOR(OpenGLNormalizePlanarYUV, GLNormPlanarYUVOp<float16_t>);
OPERATOR_SCHEMA(OpenGLNormalizePlanarYUV).NumInputs(3).NumOutputs(1);

} // namespace caffe2
