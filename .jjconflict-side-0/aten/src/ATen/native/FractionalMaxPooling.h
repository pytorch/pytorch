#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/PoolingChecks.h>
#include <c10/util/irange.h>

namespace at::native {

template<typename scalar_t>
inline std::vector<int64_t> generate_intervals(
    scalar_t sample,
    int64_t inputSize,
    int64_t outputSize,
    int64_t poolSize) {
  std::vector<int64_t> sequence(outputSize);
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

} // namespace at::native
