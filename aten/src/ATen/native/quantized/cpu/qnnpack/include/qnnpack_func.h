#pragma once
#include <conv_utils.h>

namespace qnnpack {
class PrePackConvWeights final {
 public:
  PrePackConvWeights(const conv_param_t& conv_param, const uint8_t* kernel, const int32_t* bias);

  void* getPackedWeights() const
  {
    return packed_weights_;
  }

  int64_t getOutputChannels() const
  {
    return output_channels_;
  }

  ~PrePackConvWeights()
  {
    if (packed_weights_ != nullptr) {
      free(packed_weights_);
    }
  }

  PrePackConvWeights() = delete;
  PrePackConvWeights(const PrePackConvWeights&) = delete;
  PrePackConvWeights& operator=(const PrePackConvWeights&) = delete;

 private:
  void* packed_weights_ = nullptr;
  int64_t output_channels_;
};

class PackBMatrix final {
 public:
  PackBMatrix(
      size_t input_channels,
      size_t output_channels,
      uint8_t kernel_zero_point,
      float kernel_scale,
      const uint8_t* kernel,
      const int32_t* bias);

  void* getPackedWeights() const
  {
    return packed_weights_;
  }

  size_t getInputChannels() const
  {
    return input_channels_;
  }

  size_t getOutputChannels() const
  {
    return output_channels_;
  }

  ~PackBMatrix()
  {
    if (packed_weights_ != nullptr) {
      free(packed_weights_);
    }
  }

  PackBMatrix() = delete;
  PackBMatrix(const PackBMatrix&) = delete;
  PackBMatrix& operator=(const PackBMatrix&) = delete;

 private:
  void* packed_weights_ = nullptr;
  size_t input_channels_;
  size_t output_channels_;
};

enum pytorch_qnnp_status qnnpackLinear(
    const size_t batch_size,
    const size_t input_channels,
    const size_t output_channels,
    const uint8_t input_zero_point,
    const float input_scale,
    const uint8_t kernel_zero_point,
    const float kernel_scale,
    const uint8_t output_zero_point,
    const float output_scale,
    const uint8_t output_min,
    const uint8_t output_max,
    const uint8_t* input,
    const size_t input_stride,
    void* packed_weights,
    uint8_t* output,
    const size_t output_stride,
    pthreadpool_t threadpool);

enum pytorch_qnnp_status qnnpackConv(
    const conv_param_t& conv_p,
    void* packed_weights,
    const size_t batch_size,
    const size_t input_height,
    const size_t input_width,
    const float input_scale,
    const uint8_t input_zero_point,
    const uint8_t* input,
    const float output_scale,
    const uint8_t output_zero_point,
    uint8_t* output,
    pthreadpool_t threadpool);

enum pytorch_qnnp_status qnnpackLinearDynamic(
    const size_t batch_size,
    const size_t input_channels,
    const size_t output_channels,
    const uint8_t input_zero_point,
    const float input_scale,
    const uint8_t kernel_zero_point,
    const float kernel_scale,
    const uint8_t* input,
    const size_t input_stride,
    void* packed_weights,
    const float* bias,
    float* output,
    const size_t output_stride,
    pthreadpool_t threadpool);

} // namespace qnnpack
