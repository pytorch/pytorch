#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <c10/util/irange.h>

namespace at::native {

template<typename scalar_t>
inline std::vector<int> generate_intervals(
    scalar_t sample,
    int64_t inputSize,
    int64_t outputSize,
    int64_t poolSize) {
  std::vector<int> sequence(outputSize);
  if (outputSize > 1) {
    scalar_t alpha = static_cast<scalar_t>(inputSize - poolSize) /
      static_cast<scalar_t>(outputSize - 1);

    for (const auto i : c10::irange(outputSize - 1)) {
      sequence[i] =
        static_cast<int>((i + sample) * alpha) - static_cast<int>(sample * alpha);
    }
  }
  if (outputSize > 0) {
    sequence[outputSize - 1] = inputSize - poolSize;
  }
  return sequence;
}

template <int64_t ndim>
inline void fractional_max_pool_check_shape(
    const Tensor& input,
    const Tensor& randomSamples) {

  TORCH_CHECK(
      input.scalar_type() == randomSamples.scalar_type(),
      "Expect _random_samples to have the same dtype as input");

  int64_t ndimension = randomSamples.ndimension();
  TORCH_CHECK(
      ndimension == 3,
      "Expect _random_samples to have 3 dimensions, got ", ndimension);

  int64_t N = randomSamples.size(0);
  int64_t C = randomSamples.size(1);
  int64_t D = randomSamples.size(2);

  int64_t input_batch, input_channel;
  if (ndim == 2) {
    // fractional_max_pool2d
    if (input.ndimension() == 3) {
      input_batch = 1;
      input_channel = input.size(0);
    } else {
      input_batch = input.size(0);
      input_channel = input.size(1);
    }
  } else {
    // factional_max_pool3d
    if (input.ndimension() == 4) {
      input_batch = 1;
      input_channel = input.size(0);
    } else {
      input_batch = input.size(0);
      input_channel = input.size(1);
    }
  }

  TORCH_CHECK(
      N >= input_batch,
      "Expect _random_samples.size(0) no less then input batch size.");
  TORCH_CHECK(
      C == input_channel,
      "Expect _random_samples.size(1) equals to input channel size.");
  TORCH_CHECK(
      D == ndim,
      "Expect _random_samples.size(2) equals to ", ndim, "; got ", D, ".");
}

} // namespace at::native
