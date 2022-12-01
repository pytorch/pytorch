#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <c10/util/irange.h>

namespace at { namespace native {

template<typename scalar_t>
static inline std::vector<int> generate_intervals(
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
static inline void fractional_max_pool_check_shape(
    const Tensor& input,
    const Tensor& randomSamples) {

  TORCH_CHECK(
      input.scalar_type() == randomSamples.scalar_type(),
      "Expect _random_samples to have the dtype as input");

  int64_t ndimension = randomSamples.ndimension();
  TORCH_CHECK(
      ndimension == 3,
      "Expect _random_samples to have 3 dimensions, got ", ndimension);

  int64_t N = randomSamples.size(0);
  int64_t C = randomSamples.size(1);
  int64_t D = randomSamples.size(2);

  int64_t input_channel = ndim == 3 ? input.size(-4) : input.size(-3);
  TORCH_CHECK(
      C == input_channel,
      "Expect _random_samples.size(1) equals to input channel size.");
  TORCH_CHECK(
      N > 0 && C > 0 && D == ndim,
      "Expect _random_samples to in shape of {nbatch, channels, ", ndim,
      "}; nbatch and channels must be positive, got: ",
      "{", N, ", ", C, ", ", D, "}.");
}

}} // at::native
