#pragma once

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

// beta/threshold are kept at opmath (float) precision for every input dtype,
// matching the CPU and CUDA kernels.
struct SoftplusParams {
  float beta;
  float threshold;
};
