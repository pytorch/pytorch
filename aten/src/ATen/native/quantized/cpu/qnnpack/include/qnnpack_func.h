#pragma once
#include <conv_utils.h>

namespace qnnpack {

class PackBMatrix final {
 public:
  PackBMatrix(
      size_t input_channels,
      size_t output_channels,
      uint8_t kernel_zero_point,
      float kernel_scale,
      const uint8_t* kernel,
      const int32_t* bias);

  void* getPackedWeights()
  {
    return packed_weights_;
  }

  size_t getInputChannels()
  {
    return input_channels_;
  }

  size_t getOutputChannels()
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
    size_t batch_size,
    size_t input_channels,
    size_t output_channels,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t kernel_zero_point,
    float kernel_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    const uint8_t* input,
    size_t input_stride,
    void* packed_weights,
    uint8_t* output,
    size_t output_stride,
    pthreadpool_t threadpool);

} // namespace qnnpack
