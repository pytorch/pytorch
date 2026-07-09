#pragma once

// min/max are converted to opmath precision (float) on the host, matching the
// CUDA kernel, so this struct is always instantiated at T = float.
template <typename T>
struct HardtanhBackwardParams {
  T min;
  T max;
};

template <typename T>
struct ELUParams {
  T alpha;
  T scale;
  T input_scale;
};

template <typename T>
struct ELUBackwardParams {
  T alpha;
  T scale;
  T input_scale;
  bool is_result;
};
