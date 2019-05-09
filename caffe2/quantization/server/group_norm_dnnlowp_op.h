#pragma once

#include <vector>

#include "caffe2/operators/group_norm_op.h"
#include "caffe2/quantization/server/dnnlowp_op.h"

namespace caffe2 {

using GroupNormFP32Op = GroupNormOp<float, CPUContext>;

template <typename T>
class GroupNormDNNLowPOp final : public DNNLowPOp<T, GroupNormFP32Op> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, GroupNormFP32Op);

  GroupNormDNNLowPOp(const OperatorDef& operator_def, Workspace* ws);

  bool RunOnDevice() override;

 private:
  bool GetQuantizationParameters();

  void QuantizeGamma();

  void QuantizeGammaImpl();

  void QuantizeBeta();

  bool RunOnDeviceWithOrderNCHW();

  bool RunOnDeviceWithOrderNHWC();

  void QuantizedGroupMomentsNCHW(
      int N,
      int G,
      int K,
      int HxW,
      const T* X,
      int32_t* mu,
      int32_t* rsig);

  void QuantizedGroupMomentsNHWC(
      int N,
      int G,
      int K,
      int HxW,
      const T* X,
      int32_t* mu,
      int32_t* rsig);

  void DequantizedGroupMomentsNCHW(
      int N,
      int G,
      int K,
      int HxW,
      const T* X,
      float* mu,
      float* rsig);

  void DequantizedGroupMomentsNHWC(
      int N,
      int G,
      int K,
      int HxW,
      const T* X,
      float* mu,
      float* rsig);

  void ComputeQuantizedInvStd(
      int N,
      const float* var,
      float* rsig,
      int32_t* rsig_quantized);

  void ComputeQuantizedFusedParams(
      int N,
      int G,
      int K,
      const int32_t* mu,
      const int32_t* rsig,
      const int32_t* gamma,
      const int32_t* beta,
      int32_t* scale,
      int32_t* bias);

  void ComputeDequantizedFusedParams(
      int N,
      int G,
      int K,
      const float* mu,
      const float* rsig,
      const float* gamma,
      const float* beta,
      float* scale,
      float* bias);

  void AffineBatchChannelQuantizedNCHW(
      int N,
      int C,
      int HxW,
      const T* X,
      const int32_t* scale,
      const int32_t* bias,
      T* Y);

  void AffineBatchChannelQuantizedNHWC(
      int N,
      int C,
      int HxW,
      const T* X,
      const int32_t* scale,
      const int32_t* bias,
      T* Y);

  void AffineBatchChannelDequantizedNCHW(
      int N,
      int C,
      int HxW,
      const float* X,
      const float* scale,
      const float* bias,
      float* Y);

  void AffineBatchChannelDequantizedNHWC(
      int N,
      int C,
      int HxW,
      const float* X,
      const float* scale,
      const float* bias,
      float* Y);

  const bool is_test_;
  const int group_;
  const float epsilon_;
  const StorageOrder order_;
  const bool is_param_constant_;

  std::vector<int32_t> mu_quantized_;
  std::vector<int32_t> rsig_quantized_;
  std::vector<float> mu_dequantized_;
  std::vector<float> rsig_dequantized_;
  dnnlowp::TensorQuantizationParams rsig_qparams_;

  std::vector<int32_t> gamma_quantized_;
  std::vector<int32_t> beta_quantized_;
  std::vector<float> gamma_dequantized_;
  std::vector<float> beta_dequantized_;
  const int32_t* gamma_quantized_data_ = nullptr;
  const int32_t* beta_quantized_data_ = nullptr;
  const float* gamma_dequantized_data_ = nullptr;
  const float* beta_dequantized_data_ = nullptr;

  std::vector<int32_t> scale_quantized_;
  std::vector<int32_t> bias_quantized_;
  std::vector<float> scale_dequantized_;
  std::vector<float> bias_dequantized_;
  dnnlowp::TensorQuantizationParams internal_qparams_;

  std::vector<float> X_dequantized_;
  std::vector<int32_t> Y_int32_;

  float cached_X_qparams_scale_ = 0.0f;

  // Input: X, gamma, beta
  // Output: Y, mu, inv_sig
  INPUT_TAGS(INPUT, GAMMA, BETA);
  OUTPUT_TAGS(OUTPUT, MU, INV_SIGMA);
};

namespace internal {

template <typename T>
void VectorMomentsAVX2(const int N, const T* src, int64_t* sum, int64_t* sumsq);

void ComputeQuantizedFusedParamsAVX2(
    const int N,
    const int G,
    const int K,
    const int32_t X_zero_point,
    const int32_t* mu,
    const int32_t* rsig,
    const int32_t* gamma,
    int32_t* scale,
    int32_t* bias);

template <typename T>
void AffineBatchChannelAndRequantizeNCHWAVX2(
    const int N,
    const int C,
    const int HxW,
    const dnnlowp::RequantizationParams& params,
    const T* X,
    const int32_t* scale,
    const int32_t* bias,
    T* Y);

template <typename T>
void AffineBatchChannelAndRequantizeNHWCAVX2(
    const int N,
    const int C,
    const int HxW,
    const dnnlowp::RequantizationParams& params,
    const T* X,
    const int32_t* scale,
    const int32_t* bias,
    T* Y);

} // namespace internal

} // namespace caffe2
