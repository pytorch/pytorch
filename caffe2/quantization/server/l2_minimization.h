#pragma once

#include "caffe2/quantization/server/quantization_error_minimization.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

namespace dnnlowp {

/**
 * A quantization scheme that minimizes L2 norm of quantization error.
 */
class L2ErrorMinimization : public NormMinimization {
 public:
  L2ErrorMinimization() : NormMinimization(L2){};
};

namespace internal {

float L2MinimizationKernelAVX2(
    int precision,
    float* bins,
    int nbins,
    float bin_width,
    float dst_bin_width,
    int start_bin);

} // namespace internal

} // namespace dnnlowp
