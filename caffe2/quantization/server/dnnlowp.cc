#include "dnnlowp.h"
#include "caffe2/core/logging.h"
#include "dnnlowp_op.h"
#include "kl_minimization.h"
#include "l2_minimization.h"

#include <cassert>
#include <cctype>
#ifdef _OPENMP
#include <omp.h>
#endif

C10_DEFINE_int32(
    caffe2_dnnlowp_activation_quantization_precision,
    8,
    "Precision used for activation tensors");
C10_DEFINE_int32(
    caffe2_dnnlowp_weight_quantization_precision,
    8,
    "Precision used for weight tensors");
C10_DEFINE_int32(
    caffe2_dnnlowp_requantization_multiplier_precision,
    32,
    "Precision of integer multipliers used for rescaling quantized numbers");
C10_DEFINE_int32(
    caffe2_dnnlowp_eltwise_quantization_precision,
    16,
    "Precision used for intermediate numbers during elementwise operations");
C10_DEFINE_bool(
    caffe2_dnnlowp_force_scale_power_of_two,
    false,
    "When true, force quantization scales to a power of two");
C10_DEFINE_bool(
    caffe2_dnnlowp_preserve_activation_sparsity,
    false,
    "When true, 0 is mapped to 0 after quantization: "
    "i.e., symmetric quantization");
C10_DEFINE_bool(
    caffe2_dnnlowp_preserve_weight_sparsity,
    false,
    "When true, 0 is mapped to 0 after quantization: "
    "i.e., symmetric quantization");
C10_DEFINE_string(
    caffe2_dnnlowp_activation_quantization_kind,
    "min_max",
    "Quantization method for activation tensors. "
    "Allowed values: min_max, l2, l2_approx, kl, l1, p99");
C10_DEFINE_string(
    caffe2_dnnlowp_weight_quantization_kind,
    "min_max",
    "Quantization method for weight tensors. "
    "Allowed values: min_max, l2, l2_approx, kl, l1, p99");
C10_DEFINE_int32(
    caffe2_dnnlowp_nbits_in_non_outlier,
    8,
    "When outlier-aware quantization is used, if a quantized number can be "
    "represented by this number of bits, it is considered not an outlier so "
    "handled with 16-bit accumulation");
C10_DEFINE_int32(
    caffe2_dnnlowp_copy_to_32bit_frequency,
    32,
    "When outlier-aware quantization is used, this option specifies how often "
    "we spill 16-bit accumulated numbers to 32-bit during the first pass");
C10_DEFINE_bool(
    caffe2_dnnlowp_force_slow_path,
    false,
    "When true, use slow path in quantization");

namespace dnnlowp {

using namespace std;

QuantizationFactory::QuantizationKind StringToKind(const string& s) {
  string s_lower(s);
  transform(s_lower.begin(), s_lower.end(), s_lower.begin(), ::tolower);

  if (s_lower == "min_max") {
    return QuantizationFactory::MIN_MAX_QUANTIZATION;
  } else if (s_lower == "l1") {
    return QuantizationFactory::L1_MIN_QUANTIZATION;
  } else if (s_lower == "l2") {
    return QuantizationFactory::L2_MIN_QUANTIZATION;
  } else if (s_lower == "l2_approx") {
    if (FLAGS_caffe2_dnnlowp_preserve_weight_sparsity ||
        FLAGS_caffe2_dnnlowp_preserve_activation_sparsity) {
      return QuantizationFactory::L2_MIN_QUANTIZATION;
    } else {
      return QuantizationFactory::L2_MIN_QUANTIZATION_APPROX;
    }
  } else if (s_lower == "kl") {
    return QuantizationFactory::KL_MIN_QUANTIZATION;
  } else if (s_lower == "p99") {
    return QuantizationFactory::P99_QUANTIZATION;
  } else {
    assert(false);
    return QuantizationFactory::MIN_MAX_QUANTIZATION;
  }
}

QuantizationFactory* QuantizationFactory::GetDefaultInstance() {
  static QuantizationFactory singleton(
      FLAGS_caffe2_dnnlowp_activation_quantization_precision,
      FLAGS_caffe2_dnnlowp_weight_quantization_precision,
      FLAGS_caffe2_dnnlowp_requantization_multiplier_precision,
      FLAGS_caffe2_dnnlowp_eltwise_quantization_precision,
      FLAGS_caffe2_dnnlowp_preserve_activation_sparsity,
      FLAGS_caffe2_dnnlowp_preserve_weight_sparsity,
      FLAGS_caffe2_dnnlowp_force_scale_power_of_two,
      StringToKind(FLAGS_caffe2_dnnlowp_activation_quantization_kind),
      StringToKind(FLAGS_caffe2_dnnlowp_weight_quantization_kind));

  static bool log_printed = false;
  if (!log_printed) {
    LOG(INFO) << "activation_precision "
              << FLAGS_caffe2_dnnlowp_activation_quantization_precision;
    LOG(INFO) << "weight_precision "
              << FLAGS_caffe2_dnnlowp_weight_quantization_precision;
    LOG(INFO) << "requantization_multiplier_precision "
              << FLAGS_caffe2_dnnlowp_requantization_multiplier_precision;
    LOG(INFO) << "eltwise_quantize_precision "
              << FLAGS_caffe2_dnnlowp_eltwise_quantization_precision;
    LOG(INFO) << "preserve_activation_sparsity "
              << FLAGS_caffe2_dnnlowp_preserve_activation_sparsity;
    LOG(INFO) << "preserve_weight_sparsity "
              << FLAGS_caffe2_dnnlowp_preserve_weight_sparsity;
    LOG(INFO) << "force_scale_power_of_two "
              << FLAGS_caffe2_dnnlowp_force_scale_power_of_two;
    LOG(INFO) << "activation_quantization_kind "
              << FLAGS_caffe2_dnnlowp_activation_quantization_kind;
    LOG(INFO) << "weight_quantization_kind "
              << FLAGS_caffe2_dnnlowp_weight_quantization_kind;
    LOG(INFO) << "nbits_in_non_outlier "
              << FLAGS_caffe2_dnnlowp_nbits_in_non_outlier;
    LOG(INFO) << "copy_to_32bit_frequency "
              << FLAGS_caffe2_dnnlowp_copy_to_32bit_frequency;
    LOG(INFO) << "omp_get_max_threads() " << caffe2::dnnlowp_get_max_threads();

    log_printed = true;
  }

  return &singleton;
}

QuantizationFactory::QuantizationFactory(
    int activation_precision,
    int weight_precision,
    int requantization_multiplier_precision,
    int eltwise_quantize_precision,
    bool preserve_activation_sparsity,
    bool preserve_weight_sparsity,
    bool force_scale_power_of_two,
    QuantizationKind activation_kind,
    QuantizationKind weight_kind)
    : activation_precision_(activation_precision),
      weight_precision_(weight_precision),
      requantization_multiplier_precision_(requantization_multiplier_precision),
      eltwise_quantize_precision_(eltwise_quantize_precision),
      preserve_activation_sparsity_(preserve_activation_sparsity),
      preserve_weight_sparsity_(preserve_weight_sparsity),
      force_scale_power_of_two_(force_scale_power_of_two),
      activation_kind_(activation_kind),
      weight_kind_(weight_kind) {}

TensorQuantizationParams QuantizationFactory::ChooseQuantizationParams(
    const Histogram& hist,
    QuantizationKind kind,
    int precision,
    bool preserve_sparsity) const {
  switch (kind) {
    case L2_MIN_QUANTIZATION:
      return L2ErrorMinimization().ChooseQuantizationParams(
          hist, preserve_sparsity, precision);
    case L2_MIN_QUANTIZATION_APPROX:
      return L2ErrorMinimization().NonlinearQuantizationParamsSearch(
          hist, preserve_sparsity, precision);
    case L1_MIN_QUANTIZATION:
      return L1ErrorMinimization().ChooseQuantizationParams(
          hist, preserve_sparsity, precision);
    case KL_MIN_QUANTIZATION:
      return KLDivergenceMinimization().ChooseQuantizationParams(
          hist, preserve_sparsity, precision);
    case P99_QUANTIZATION:
      assert(preserve_sparsity);
      return P99().ChooseQuantizationParams(hist, preserve_sparsity, precision);
    case MIN_MAX_QUANTIZATION:
    default:
      return ChooseQuantizationParams(
          hist.Min(), hist.Max(), precision, preserve_sparsity);
  }
}

TensorQuantizationParams QuantizationFactory::ChooseQuantizationParams(
    const Histogram& hist,
    bool is_weight) const {
  if (is_weight) {
    return ChooseQuantizationParams(
        hist,
        GetWeightKind(),
        GetWeightPrecision(),
        GetPreserveWeightSparsity());
  } else {
    return ChooseQuantizationParams(
        hist,
        GetActivationKind(),
        GetActivationPrecision(),
        GetPreserveActivationSparsity());
  }
}

TensorQuantizationParams QuantizationFactory::ChooseQuantizationParams(
    const float* values,
    int len,
    QuantizationKind kind,
    int precision,
    bool preserve_sparsity) const {
  float min = 0, max = 0;
  fbgemm::FindMinMax(values, &min, &max, len);

  if (MIN_MAX_QUANTIZATION == kind) {
    return ChooseQuantizationParams(min, max, precision, preserve_sparsity);
  } else {
    if (0 == len) {
      return ChooseQuantizationParams(min, max, precision, preserve_sparsity);
    }

    Histogram hist(2048, min, max);
    for (int i = 0; i < len; ++i) {
      hist.Add(values[i]);
    }

    return ChooseQuantizationParams(hist, kind, precision, preserve_sparsity);
  }
}

TensorQuantizationParams QuantizationFactory::ChooseQuantizationParams(
    const float* values,
    int len,
    bool is_weight) const {
  if (is_weight) {
    return ChooseQuantizationParams(
        values,
        len,
        GetWeightKind(),
        GetWeightPrecision(),
        GetPreserveWeightSparsity());
  } else {
    return ChooseQuantizationParams(
        values,
        len,
        GetActivationKind(),
        GetActivationPrecision(),
        GetPreserveActivationSparsity());
  }
}

RequantizationParams QuantizationFactory::ChooseRequantizationMultiplier(
    float real_multiplier,
    TensorQuantizationParams target_qparams) const {
  RequantizationParams params;
  params.target_qparams = target_qparams;
  params.real_multiplier = real_multiplier;

  fbgemm::ChooseRequantizationMultiplier(
      real_multiplier,
      &params.multiplier,
      &params.right_shift,
      requantization_multiplier_precision_);

  return params;
}

} // namespace dnnlowp
