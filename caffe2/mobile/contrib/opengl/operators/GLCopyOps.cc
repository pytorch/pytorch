
#include "caffe2/core/common.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"

#include "../core/DataTransfer.h"
#include "../core/GLContext.h"
#include "../core/GLImage.h"
#include "../core/GLPlainTexture.h"
#include "../core/ImageAllocator.h"

#include <algorithm>

namespace caffe2 {
template <class T>
class CopyToOpenGLOp final : public Operator<CPUContext>, ImageAllocator<T> {
 public:
  CopyToOpenGLOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    // caffe2::Timer timer;
    const TensorCPU& X = Input(0);
    const int num_images = X.dim32(0);
    const int input_channels = X.dim32(1);
    const int input_width = X.dim32(3);
    const int input_height = X.dim32(2);
    const int input_size = input_width * input_height;

    // set up the OpenGL context
    GLContext::getGLContext()->set_context();

    const float* input = X.template data<float>();

    int tile_x = GetSingleArgument<int>("tile_x", 1);
    int tile_y = GetSingleArgument<int>("tile_y", 1);

    GLImageVector<T>* output_image = ImageAllocator<T>::newImage(num_images,
                                                                 input_width,
                                                                 input_height,
                                                                 input_channels,
                                                                 tile_x,
                                                                 tile_y,
#if CAFFE2_IOS
                                                                 true
#else
                                                                 false
#endif
    );

    if (output_image->tile_x() > 1 || output_image->tile_y() > 1) {
      LOG(INFO) << "CopyToOpenGLOp tiling: " << output_image->tile_x() << ":"
                << output_image->tile_y();
    }

    Outputs()[0]->Reset(output_image);

    for (int i = 0; i < num_images; i++) {
      const auto textures = (*output_image)[i]->textures;
      for (int slice = 0; slice < textures.size(); slice++) {
        // timer.Start();

        textures[slice]->map_load([&](void* buffer,
                                      size_t width,
                                      size_t height,
                                      size_t stride,
                                      size_t channels,
                                      const GLTexture::Type& type) {
          for (int y = 0; y < tile_y; y++) {
            for (int x = 0; x < tile_x; x++) {
              const int tiles = slice * tile_x * tile_y + y * tile_x + x;
              const int slice_channels = std::min(4, input_channels - 4 * tiles);
              interleaveSlice(
                  (float16_t*)buffer + 4 * (y * input_height * stride + x * input_width),
                  &input[i * input_channels * input_size + 4 * tiles * input_size],
                  input_width,
                  input_height,
                  stride, // texture stride
                  slice_channels);
            }
          }
        });
        // LOG(INFO) << "Texture uploading takes " << timer.MilliSeconds() << " ms";
      }
    }

    return true;
  }
};

REGISTER_CPU_OPERATOR(CopyToOpenGL, CopyToOpenGLOp<float16_t>);
OPERATOR_SCHEMA(CopyToOpenGL).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});

template <class T>
class CopyFromOpenGLOp final : public Operator<CPUContext> {
 public:
  CopyFromOpenGLOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    caffe2::Timer timer;
    const GLImageVector<T>& X = Inputs()[0]->template Get<GLImageVector<T>>();
    const int num_images = X.size();
    const int input_channels = X.channels();
    const int input_width = X.width();
    const int input_height = X.height();

    TensorCPU* Y = Output(0);
    Y->Resize(num_images, input_channels, input_height, input_width);
    const int output_width = input_width;
    const int output_height = input_height;
    const int output_size = input_width * input_height;

    float* output = Y->mutable_data<float>();

    const int tile_x = X.tile_x();
    const int tile_y = X.tile_y();
    for (int i = 0; i < num_images; i++) {
      for (int slice = 0; slice < X[i]->slices; slice++) {
        timer.Start();
        const GLTexture* texture = X[i]->textures[slice];

        texture->map_read([&](const void* buffer,
                              size_t width,
                              size_t height,
                              size_t stride,
                              size_t channels,
                              const GLTexture::Type& type) {
          //#if CAFFE2_ANDROID && defined(__ARM_NEON__)
          //        if (static_cast<AndroidGLContext*>(GLContext::getGLContext())->get_platform() ==
          //        Mali) {
          //          caffe2::Timer timer;
          //          timer.Start();
          //          float16_t* copy_buffer = (float16_t*)malloc(_capacity);
          //          arm_memcpy(
          //              (volatile unsigned char*)copy_buffer, (volatile unsigned char*)buffer,
          //              _capacity);
          //          deInterleaveSlice(
          //              output + 4 * slice * output_size, copy_buffer, width, height, stride,
          //              slice_channels);
          //          free(copy_buffer);
          //          LOG(INFO) << "memcpy takes " << timer.MilliSeconds() << " ms";
          //        } else
          //#endif
          {
            gl_log(GL_VERBOSE,
                   "calling deInterleaveSlice width: %d, height: %d, stride: %d, channels: %d\n",
                   width,
                   height,
                   stride,
                   channels);

            for (int y = 0; y < tile_y; y++) {
              for (int x = 0; x < tile_x; x++) {
                const int tiles = slice * tile_x * tile_y + y * tile_x + x;
                const int slice_channels = std::min(4, input_channels - 4 * tiles);
                deInterleaveSlice(
                    output + i * input_channels * output_size + 4 * tiles * output_size,
                    (float16_t*)buffer + 4 * (y * input_height * stride + x * input_width),
                    input_width,
                    input_height,
                    stride,
                    slice_channels);
              }
            }
          }
        });
      }
    }
    return true;
  }
};

REGISTER_CPU_OPERATOR(CopyFromOpenGL, CopyFromOpenGLOp<float16_t>);
OPERATOR_SCHEMA(CopyFromOpenGL).NumInputs(1).NumOutputs(1).AllowInplace({{0, 0}});
} // namespace caffe2
