#pragma once

#include "caffe2/operators/spatial_batch_norm_op.h"
#include "caffe2/quantization/server/dnnlowp_op.h"

namespace caffe2 {

/**
 * Note this implementation assumes SCALE, BIAS, EST_MEAN, and EST_VAR inputs
 * are still in fp32, so is epsilon argument
 */
template <typename T, bool ReluFused = false>
class SpatialBNDNNLowPOp final : public DNNLowPOp<T, SpatialBNOp<CPUContext>> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, SpatialBNOp<CPUContext>);
  SpatialBNDNNLowPOp(const OperatorDef& operator_def, Workspace* ws);

  virtual ~SpatialBNDNNLowPOp() override = default;

  bool RunOnDevice() override;

 private:
  void ComputeFusedParam_(
      const int C,
      const float* scale,
      const float* bias,
      const float* mean,
      const float* var,
      float* alpha,
      float* beta);

  double epsilon_;
  const StorageOrder order_;

  Tensor alpha_;
  Tensor beta_;

  INPUT_TAGS(INPUT, SCALE, BIAS, EST_MEAN, EST_VAR);
  OUTPUT_TAGS(OUTPUT);
};

namespace internal {

template <typename T>
void SpatialBNNHWCAVX2(
    const int N,
    const int C,
    const int HxW,
    const int in_zero_point,
    const int out_zero_point,
    const T* X,
    const float* alpha,
    const float* beta,
    T* Y,
    bool relu_fused);

} // namespace internal
} // namespace caffe2
