#pragma once

#include "../core/GLFilter.h"
#include "../core/GLImage.h"
#include "gl_tiling_utils.h"

class GLConvolution : public GLFilter {
 public:
  static constexpr int MaxInputBatchSize = 8;
  static constexpr int MaxOutputBatchSize = 4;

  struct descriptor {
    int input_channels;
    int output_channels;
    point kernel_size;
    point input_tile_size;
    point output_tile_size;
    point input_tile_grid_size;
    point output_tile_grid_size;
    point input_padding;
    point input_stride;
    bool transposed;
  };

  const float* kernel;
  const float* bias;
  const float* prelu_scale;

  binding* inputData[MaxInputBatchSize];
  binding* previousData[MaxOutputBatchSize];
  binding* outputSize;
  binding* accumulate;
  binding* fusePRelu;
  binding* kernel_block[MaxInputBatchSize];
  binding* bias_block;
  binding* prelu_scale_block;
  binding* inputTileRange;

  const descriptor geometry;
  const int prelu_scale_size;
  const int input_batch_size;
  const int output_batch_size;
  const int input_tiles;
  const int output_tiles;
  const int input_tile_chunk_size;
  const int output_tile_chunk_size;
  const int input_tile_batch_size;
  const int output_tile_batch_size;
  const bool tiling;

  static const char* fragment_shader;

  GLConvolution(
      const descriptor& _geometry,
      const float* _kernel,
      const float* _bias,
      const float* _prelu_scale = nullptr,
      int _prelu_scale_size = 0,
      int _input_batch_size = 1,
      int _output_batch_size = 1,
      int _input_tiles = 1,
      int _output_tiles = 1,
      int _input_tile_chunk_size = 1,
      int _output_tile_chunk_size = 1,
      int _input_tile_batch_size = 1,
      int _output_tile_batch_size = 1,
      bool _tiling = false)
      : GLFilter(
            "GLConvolution",
            vertex_shader,
            fragment_shader,
            input_bindings(_input_batch_size, _output_batch_size),
            uniform_blocks_bindings(
                _input_batch_size,
                _output_batch_size,
                _output_tile_batch_size,
                _prelu_scale != nullptr),
            {/* no attributes */},
            {{"KERNEL_SIZE_X", caffe2::to_string(_geometry.kernel_size.x)},
             {"KERNEL_SIZE_Y", caffe2::to_string(_geometry.kernel_size.y)},
             {"INPUT_BATCH_SIZE", caffe2::to_string(_input_batch_size)},
             {"OUTPUT_BATCH_SIZE", caffe2::to_string(_output_batch_size)},
             {"INPUT_TILES", caffe2::to_string(_input_tiles)},
             {"OUTPUT_TILES", caffe2::to_string(_output_tiles)},
             {"INPUT_TILE_WIDTH",
              caffe2::to_string(_geometry.input_tile_size.x)},
             {"INPUT_TILE_HEIGHT",
              caffe2::to_string(_geometry.input_tile_size.y)},
             {"OUTPUT_TILE_WIDTH",
              caffe2::to_string(_geometry.output_tile_size.x)},
             {"OUTPUT_TILE_HEIGHT",
              caffe2::to_string(_geometry.output_tile_size.y)},
             {"INPUT_TILE_X",
              caffe2::to_string(_geometry.input_tile_grid_size.x)},
             {"OUTPUT_TILE_X",
              caffe2::to_string(_geometry.output_tile_grid_size.x)},
             {"INPUT_TILE_CHUNK_SIZE",
              caffe2::to_string(_input_tile_chunk_size)},
             {"OUTPUT_TILE_CHUNK_SIZE",
              caffe2::to_string(_output_tile_chunk_size)},
             {"OUTPUT_TILE_BATCH_SIZE",
              caffe2::to_string(_output_tile_batch_size)},
             {"TILED_CONVOLUTION", caffe2::to_string(_tiling)},
             {"INPUT_PADDING_X",
              caffe2::to_string(
                  _geometry.transposed
                      ? _geometry.kernel_size.x - 1 - _geometry.input_padding.x
                      : _geometry.input_padding.x)},
             {"INPUT_PADDING_Y",
              caffe2::to_string(
                  _geometry.transposed
                      ? _geometry.kernel_size.y - 1 - _geometry.input_padding.y
                      : _geometry.input_padding.y)},
             {"INPUT_STRIDE_X", caffe2::to_string(_geometry.input_stride.x)},
             {"INPUT_STRIDE_Y", caffe2::to_string(_geometry.input_stride.y)},
             {"TRANSPOSED_CONVOLUTION",
              caffe2::to_string(_geometry.transposed)},
             {"BOUNDS_CHECK_MODE",
              caffe2::to_string(bounds_check_mode(_tiling, _geometry))}}),
        kernel(_kernel),
        bias(_bias),
        prelu_scale(_prelu_scale),
        geometry(_geometry),
        prelu_scale_size(_prelu_scale_size),
        input_batch_size(_input_batch_size),
        output_batch_size(_output_batch_size),
        input_tiles(_input_tiles),
        output_tiles(_output_tiles),
        input_tile_chunk_size(_input_tile_chunk_size),
        output_tile_chunk_size(_output_tile_chunk_size),
        input_tile_batch_size(_input_tile_batch_size),
        output_tile_batch_size(_output_tile_batch_size),
        tiling(_tiling) {}

  ~GLConvolution() {}

  template <typename T>
  void convolution(
      const GLImageVector<T>& input_images,
      const GLImageVector<T>& output_images);

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

    int input_padding_x = geometry.input_padding.x,
        input_padding_y = geometry.input_padding.y;
    if (geometry.transposed) {
      input_padding_x = geometry.kernel_size.x - 1 - input_padding_x;
      input_padding_y = geometry.kernel_size.y - 1 - input_padding_y;
    }

    if (GLContext::getGLContext()->GL_EXT_texture_border_clamp_defined() ||
        (input_padding_x == 0 && input_padding_y == 0)) {
      return 0;
    } else {
      return 1;
    }
  }

  const std::vector<binding*> input_bindings(
      int input_batch_size,
      int output_batch_size) {
    std::vector<binding*> bindings({BINDING(outputSize),
                                    BINDING(accumulate),
                                    BINDING(fusePRelu),
                                    BINDING(inputTileRange)});

    for (int i = 0; i < input_batch_size; i++) {
      bindings.push_back(
          inputData[i] =
              new binding{"inputData[" + caffe2::to_string(i) + "]"});
    }

    for (int i = 0; i < output_batch_size; i++) {
      bindings.push_back(
          previousData[i] =
              new binding{"previousData[" + caffe2::to_string(i) + "]"});
    }

    return bindings;
  }

  const std::vector<binding*> uniform_blocks_bindings(
      int input_batch_size,
      int output_batch_size,
      int output_tile_batch_size,
      bool fuse_prelu) {
    std::vector<binding*> bindings({BINDING(bias_block)});
    if (fuse_prelu) {
      bindings.push_back(BINDING(prelu_scale_block));
    }

    for (int i = 0; i < std::max(input_batch_size, output_tile_batch_size);
         i++) {
      bindings.push_back(
          kernel_block[i] =
              new binding{"Kernel_block[" + caffe2::to_string(i) + "]"});
    }

    return bindings;
  }

  void pack_kernel_data_for_bached_conv(
      float16_t* data,
      size_t size,
      int input_channels,
      int output_channels,
      int is,
      int os,
      int ib);

  void pack_kernel_data_for_tiled_conv(
      float16_t* data, // destination
      size_t size,
      int input_channels,
      int output_channels,
      point input_tile_range,
      point output_tile_range);

  template <typename T>
  void run_batched_conv(
      const GLImageVector<T>& input_images,
      const GLImageVector<T>& output_images);

  template <typename T>
  void run_tiled_conv(
      const GLImageVector<T>& input_images,
      const GLImageVector<T>& output_images);
};
