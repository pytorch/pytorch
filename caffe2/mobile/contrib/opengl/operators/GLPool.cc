
#include "../core/GLFilter.h"
#include "../core/GLImage.h"
#include "../core/ImageAllocator.h"

#include "caffe2/core/timer.h"
#include "caffe2/operators/pool_op.h"

class GLPool : public GLFilter {
 public:
  typedef enum { AveragePool, MaxPool } PoolType;

  struct point {
    int x;
    int y;
  };

  struct descriptor {
    int channels;
    point kernel_size;
    point input_padding;
    point input_stride;
    point input_tile_size;
    point output_tile_size;
  };

  binding* inputData;
  binding* kernelSize;
  binding* outputSize;

  const descriptor geometry;

  GLPool(const descriptor& _geometry, PoolType poolType, bool _tiling)
      : GLFilter(
            "GLPool",
            vertex_shader,
            fragment_shader,
            {
                BINDING(inputData),
                BINDING(kernelSize),
                BINDING(outputSize),
            },
            {/* no uniform blocks */},
            {/* no attributes */},
            {{"KERNEL_SIZE_X", caffe2::to_string(_geometry.kernel_size.x)},
             {"KERNEL_SIZE_Y", caffe2::to_string(_geometry.kernel_size.y)},
             {"INPUT_PADDING_X", caffe2::to_string(_geometry.input_padding.x)},
             {"INPUT_PADDING_Y", caffe2::to_string(_geometry.input_padding.y)},
             {"INPUT_STRIDE_X", caffe2::to_string(_geometry.input_stride.x)},
             {"INPUT_STRIDE_Y", caffe2::to_string(_geometry.input_stride.y)},
             {"INPUT_TILE_WIDTH",
              caffe2::to_string(_geometry.input_tile_size.x)},
             {"INPUT_TILE_HEIGHT",
              caffe2::to_string(_geometry.input_tile_size.y)},
             {"OUTPUT_TILE_WIDTH",
              caffe2::to_string(_geometry.output_tile_size.x)},
             {"OUTPUT_TILE_HEIGHT",
              caffe2::to_string(_geometry.output_tile_size.y)},
             {"TILED_POOLING", caffe2::to_string(_tiling)},
             {"MAX_POOL", caffe2::to_string(poolType == MaxPool)},
             {"BOUNDS_CHECK_MODE", caffe2::to_string(1)}}),
        geometry(_geometry) {}
  ~GLPool() {}

  void pool(const GLImageVector<float16_t>& input_images,
            const GLImageVector<float16_t>& output_images) {
    for (int i = 0; i < input_images.size(); i++) {
      auto input_image = input_images[i];
      auto output_image = output_images[i];
      int input_slices = input_image->slices;
      int output_slices = output_image->slices;

      for (int is = 0; is < input_slices; is++) {
        run({{input_image->textures[is], inputData}},
            {output_image->textures[is]},
            [&]() {
              glUniform2i(outputSize->location, output_image->texture_width, output_image->texture_height);
              glUniform2i(kernelSize->location, geometry.kernel_size.x, geometry.kernel_size.y);
            },
            output_image->texture_width,
            output_image->texture_height);
      }
    }
  }

 private:
  /*
   * Computes BOUNDS_CHECK_MODE for the convolution parameters.
   *
   * @retval 0 if bounds check can be skipped
   * @retval non-zero if bounds check can not be skipped
   */
  inline static int bounds_check_mode(bool tiling, const descriptor& geometry) {
    if (tiling) {
      return 1;
    }

    if (GLContext::getGLContext()->GL_EXT_texture_border_clamp_defined() ||
        (geometry.input_padding.x == 0 && geometry.input_padding.y == 0)) {
      return 0;
    } else {
      return 1;
    }
  }

  static const char* fragment_shader;
};

// MARK: GLSL
const char* GLPool::fragment_shader = R"GLSL(#version 300 es
#define TILED_POOLING           $(TILED_POOLING)
#define MAX_POOL                $(MAX_POOL)

// tiling
#define INPUT_TILE_WIDTH            $(INPUT_TILE_WIDTH)
#define INPUT_TILE_HEIGHT           $(INPUT_TILE_HEIGHT)
#define OUTPUT_TILE_WIDTH           $(OUTPUT_TILE_WIDTH)
#define OUTPUT_TILE_HEIGHT          $(OUTPUT_TILE_HEIGHT)

#define BOUNDS_CHECK_MODE           $(BOUNDS_CHECK_MODE)

precision mediump float;
precision mediump int;

in highp vec2 v_texCoord;

const ivec2 input_padding = ivec2($(INPUT_PADDING_X), $(INPUT_PADDING_Y));
const ivec2 input_stride = ivec2($(INPUT_STRIDE_X), $(INPUT_STRIDE_Y));
const ivec2 kernel_size = ivec2($(KERNEL_SIZE_X), $(KERNEL_SIZE_Y));

uniform ivec2 kernelSize;
uniform ivec2 outputSize;

TEXTURE_INPUT(inputData);
TEXTURE_OUTPUT(0, outputData);

#if BOUNDS_CHECK_MODE == 0
  #define IN_BOUNDS(p, p0, p1) (true)
#else
  #define IN_BOUNDS(p, p0, p1) (all(greaterThanEqual(p, p0)) && all(lessThan(p, p1)))
#endif

// MIN_FLOAT is -2^14, which is the minimum precision requirement for mediump in OpenGL ES 3.0
const float MIN_FLOAT = -exp2(14.0);

#if TILED_POOLING

const ivec2 inputTileSize = ivec2(INPUT_TILE_WIDTH, INPUT_TILE_HEIGHT);
const ivec2 outputTileSize = ivec2(OUTPUT_TILE_WIDTH, OUTPUT_TILE_HEIGHT);

// tiled pooling
#if MAX_POOL

#define POOL { \
  pool = vec4(MIN_FLOAT); \
  for (int y = 0; y < kernelSize.y; y++) { \
    for (int x = 0; x < kernelSize.x; x++) { \
      ivec2 idx = tileCoord + ivec2(x, y); \
      if IN_BOUNDS(idx, ivec2(0), inputTileSize) { \
        vec4 data = TEXTURE_LOAD(inputData, inputTileOffset + idx); \
        pool = max(pool, data); \
      } \
    } \
  } \
}

#else

#define POOL { \
  int count = 0; \
  for (int y = 0; y < kernelSize.y; y++) { \
    for (int x = 0; x < kernelSize.x; x++) { \
      ivec2 idx = tileCoord + ivec2(x, y); \
      if IN_BOUNDS(idx, ivec2(0), inputTileSize) { \
        vec4 data = TEXTURE_LOAD(inputData, inputTileOffset + idx); \
        pool += data;\
        count += 1; \
      } \
    } \
  } \
  pool = pool / float(count); \
}

#endif // MAX_POOL

void main() {
  ivec2 inputSize = textureSize(inputData, 0);
  ivec2 texelCoord = ivec2(v_texCoord * vec2(outputSize));

  ivec2 tile = texelCoord / outputTileSize; // 2D output tile idx
  ivec2 tileCoord = texelCoord % outputTileSize; // in-tile coordinates
  tileCoord = input_stride * tileCoord - input_padding;

  ivec2 inputTileOffset = tile * inputTileSize;

#if MAX_POOL
  vec4 pool = vec4(0);
#else
  highp vec4 pool = vec4(0);
#endif

  POOL;

  outputData = TEXTURE_STORE(pool);
}

#else

// no tiling
#if MAX_POOL

#define POOL { \
  pool = vec4(MIN_FLOAT); \
  for (int y = 0; y < kernelSize.y; y++) { \
    for (int x = 0; x < kernelSize.x; x++) { \
      ivec2 idx = texelCoord + ivec2(x, y); \
      if IN_BOUNDS(idx, ivec2(0), inputSize) { \
        vec4 data = TEXTURE_LOAD(inputData, idx); \
        pool = max(pool, data); \
      } \
    } \
  } \
}

#else

#define POOL { \
  int count = 0; \
  for (int y = 0; y < kernelSize.y; y++) { \
    for (int x = 0; x < kernelSize.x; x++) { \
      ivec2 idx = texelCoord + ivec2(x, y); \
      if IN_BOUNDS(idx, ivec2(0), inputSize) { \
        vec4 data = TEXTURE_LOAD(inputData, idx); \
        pool += data; \
        count += 1; \
      } \
    } \
  } \
  pool = pool / float(count); \
}

#endif // MAX_POOL

void main() {
  ivec2 inputSize = textureSize(inputData, 0);
  ivec2 texelCoord = input_stride * ivec2(v_texCoord * vec2(outputSize)) - input_padding;
#if MAX_POOL
  vec4 pool = vec4(0);
#else
  highp vec4 pool = vec4(0);
#endif

  POOL;

  outputData = TEXTURE_STORE(pool);
}
#endif // TILED_POOLING

)GLSL";

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

template <typename T, GLPool::PoolType poolType>
class GLPoolOp final : public ConvPoolOpBase<CPUContext>, ImageAllocator<float16_t> {
 public:
  GLPoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(order_ == StorageOrder::NCHW, "OpenGL only supports NCHW order.");
    CAFFE_ENFORCE(dilation_h() == 1 && dilation_w() == 1,
                  "Pooling op does not support dilation right now.");
    if (!global_pooling_) {
      CAFFE_ENFORCE(pad_t() < kernel_h() && pad_b() < kernel_h() && pad_l() < kernel_w() &&
                        pad_r() < kernel_w(),
                    "Pad should be smaller than kernel.");
    }
  }

  bool RunOnDeviceWithOrderNCHW() override {
    const GLImageVector<T>& input = OperatorBase::Inputs()[0]->template Get<GLImageVector<T>>();
    const int num_images = input.size();
    const int input_channels = input.channels();
    const int input_width = input.width();
    const int input_height = input.height();

    int output_height;
    int output_width;
    const int output_channels = input_channels;

    computeOutputHW(this, input_height, input_width, &output_height, &output_width);

    int is_last = OperatorBase::GetSingleArgument<int>("is_last", 0);

    const int input_tile_x = input.tile_x(), input_tile_y = input.tile_y();
    const int output_tile_x = input_tile_x, output_tile_y = input_tile_y;

    GLImageVector<T>* output = ImageAllocator<T>::newImage(
        num_images, output_width, output_height, output_channels, output_tile_x, output_tile_y, is_last);

    GLPool::descriptor geometry{input_channels,
                                {kernel_w(), kernel_h()},
                                {pad_l(), pad_t()},
                                {stride_w(), stride_h()},
                                {input_width, input_height},
                                {output_height, output_width}};

    if (!glPool_) {
      LOG(INFO) << input_channels << ": " << input_height << " X " << input_width << " => " << output_channels << ": "
                << output_height << " X " << output_width << " Kernel: " << kernel_w() << "X" << kernel_h()
                << " Tiling: " << input_tile_x << "X" << input_tile_y;

      glPool_.reset(new GLPool(geometry, poolType, input_tile_x > 1 || input_tile_y > 1));
    }

    glPool_->pool(input, *output);

    OperatorBase::Outputs()[0]->Reset(output);

    return true;
  }

 private:
  std::unique_ptr<GLPool> glPool_;
};

namespace {
REGISTER_CPU_OPERATOR(OpenGLAveragePool, GLPoolOp<float16_t, GLPool::AveragePool>);
REGISTER_CPU_OPERATOR(OpenGLMaxPool, GLPoolOp<float16_t, GLPool::MaxPool>);
OPERATOR_SCHEMA(OpenGLAveragePool).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(OpenGLMaxPool).NumInputs(1).NumOutputs(1);
}; // namespace
}; // namespace caffe2
