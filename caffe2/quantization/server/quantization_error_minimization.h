#pragma once

#include "dnnlowp.h"

namespace dnnlowp {

class QuantizationErrorMinimization {
 public:
  virtual TensorQuantizationParams ChooseQuantizationParams(
      const Histogram& hist,
      bool preserve_sparsity = false,
      int precision = 8) = 0;
  virtual ~QuantizationErrorMinimization(){};
};

class NormMinimization : public QuantizationErrorMinimization {
 public:
  enum Kind {
    L1,
    L2,
  };

  NormMinimization(Kind kind) : kind_(kind) {}

  /**
   * Faster approximate search
   */
  TensorQuantizationParams NonlinearQuantizationParamsSearch(
      const Histogram& hist,
      bool preserve_sparsity = false,
      int precision = 8);

  TensorQuantizationParams ChooseQuantizationParams(
      const Histogram& hist,
      bool preserve_sparsity = false,
      int precision = 8) override;

 protected:
  Kind kind_;
};

class L1ErrorMinimization : public NormMinimization {
 public:
  L1ErrorMinimization() : NormMinimization(L1) {}
};

class P99 : public QuantizationErrorMinimization {
 public:
  float threshold_;
  P99(float p99_threshold = 0.99) : threshold_(p99_threshold) {}
  TensorQuantizationParams ChooseQuantizationParams(
      const Histogram& hist,
      bool preserve_sparsity = true,
      int precision = 8) override;
}; // class P99QuantizationFactory

} // namespace dnnlowp
