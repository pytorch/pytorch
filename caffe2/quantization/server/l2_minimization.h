#ifndef DNNLOWP_L2_MINIMIZATION_H
#define DNNLOWP_L2_MINIMIZATION_H

#include "quantization_error_minimization.h"

#include <algorithm>
#include <limits>
#include <cassert>
#include <iostream>
#include <cmath>

namespace dnnlowp {

/**
 * A quantization scheme that minimizes L2 norm of quantization error.
 */
class L2ErrorMinimization : public NormMinimization {
 public:
  L2ErrorMinimization() : NormMinimization(L2) { };
};

float L2MinimizationKernelAVX2(
    int precision,
    float* bins,
    int nbins,
    float bin_width,
    float dst_bin_width,
    int start_bin);

} // namespace dnnlowp

#endif // DNNLOWP_L2_MINIMIZATION_H
