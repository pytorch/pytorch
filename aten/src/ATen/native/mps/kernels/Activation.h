#pragma once

// Scalar params are kept at opmath (float) precision for every input dtype,
// matching the CPU and CUDA kernels.
struct ELUParams {
  float alpha;
  float scale;
  float input_scale;
};

struct ELUBackwardParams {
  float alpha;
  float scale;
  float input_scale;
  bool is_result;
};

struct SoftplusParams {
  float beta;
  float threshold;
};

struct HardtanhParams {
  float min_val;
  float max_val;
};

template <typename T>
struct ThresholdParams {
  T threshold;
  T value;
};
