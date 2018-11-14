#ifndef DNNLOWP_KL_MINIMIZATION_H
#define DNNLOWP_KL_MINIMIZATION_H

#include "quantization_error_minimization.h"

namespace dnnlowp {

/**
 * A quantization scheme that minimizes Kullback-Leiber divergence.
 */
class KLDivergenceMinimization final : public QuantizationErrorMinimization {
 public:
  TensorQuantizationParams ChooseQuantizationParams(
    const Histogram& hist, bool preserve_sparsity = false, int precision = 8)
    override;
};

} // namespace dnnlowp

#endif // DNNLOWP_KL_MINIMIZATION_H
