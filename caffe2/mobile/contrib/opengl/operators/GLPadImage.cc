
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

in highp vec2 v_texCoord;

uniform ivec2 padSize;
uniform ivec2 inputSize;
uniform ivec2 outputSize;

TEXTURE_INPUT(inputData);
TEXTURE_OUTPUT(0, outputData);

void main() {
  ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize)) - padSize;
  texelCoord = max(texelCoord, -texelCoord);
  texelCoord = min(texelCoord, ivec2(2) * (inputSize - 1) - texelCoord);
  vec4 value = TEXTURE_LOAD(inputData, texelCoord);
  outputData = TEXTURE_STORE(value);
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

template <typename OPBase>
static void computeOutputHW(OPBase* op, int H, int W, int* OH, int* OW) {
  Tensor<CPUContext> input, output;
  input.Resize(1, 1, H, W);
  op->SetOutputSize(input, &output, 1);
  CAFFE_ENFORCE_EQ(output.ndim(), 4);
  *OH = output.dim(2);
  *OW = output.dim(3);
}

template <class T>
class OpenGLPadImageOp final : public ConvPoolOpBase<CPUContext>, ImageAllocator<T> {
 public:
  OpenGLPadImageOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws),
        mode_(OperatorBase::GetSingleArgument<string>("mode", "")) {
    OPERATOR_NEEDS_FEATURE(order_ == StorageOrder::NCHW, "OpenGL only supports NCHW order.");
    OPERATOR_NEEDS_FEATURE(mode_ == "reflect", "OpenGL only supports reflection");

    CAFFE_ENFORCE(legacy_pad_ == LegacyPadding::NOTSET,
                  "Padding layer only supports explicit pad values.");
    CAFFE_ENFORCE(dilation_h() == 1 && dilation_w() == 1,
                  "Pooling op does not support dilation right now.");
    CAFFE_ENFORCE(stride_h() == 1 && stride_w() == 1,
                  "Pooling op does not support stride right now.");
    // Pad op does not use kernel sizes, so we set it to 1 for computing the
    // output size.
    kernel_.assign(pads_.size() / 2, 1);
  }

  bool RunOnDeviceWithOrderNCHW() override {
    const GLImageVector<T>& input = Inputs()[0]->template Get<GLImageVector<T>>();

    const int num_images = input.size();
    const int input_width = input.width();
    const int input_height = input.height();
    const int input_channels = input.channels();
    const int output_channels = input_channels;

    int output_height, output_width;
    computeOutputHW(this, input_height, input_width, &output_height, &output_width);

    int is_last = OperatorBase::GetSingleArgument<int>("is_last", 0);

    GLImageVector<T>* output = ImageAllocator<T>::newImage(
        num_images, output_width, output_height, output_channels, is_last);

    if (!padImage_) {
      padImage_.reset(new GLPadImage());
      LOG(INFO) << input_channels << ": " << input_height << " X " << input_width << " => "
                << output_channels << ": " << output_height << " X " << output_width;
      LOG(INFO) << "Padmode: " << mode_ << ", pad_l = " << pad_l() << ", pad_r = " << pad_r() << ", pad_t = " << pad_t()
                << ", pad_b = " << pad_b();
    }

    padImage_->pad(input, *output, pad_l(), pad_t());

    Outputs()[0]->Reset(output);

    return true;
  }

 private:
  std::string mode_;
  std::unique_ptr<GLPadImage> padImage_;
};

REGISTER_CPU_OPERATOR(OpenGLPadImage, OpenGLPadImageOp<float16_t>);
OPERATOR_SCHEMA(OpenGLPadImage).NumInputs(1).NumOutputs(1);
} // namespace caffe2
