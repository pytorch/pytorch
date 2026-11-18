#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cstdint>
#include <string>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wswitch-enum")

namespace c10 {

/**
 * QScheme is an enum that specifies the type of quantization. This has a one
 * to one correspondence with Quantizer
 * Please refer to ATen/quantized/Quantizer.h to see the Quantizers classes.
 * Keep this file in sync with torch/nn/_qscheme.py
 */
enum class QScheme : uint8_t {
  PER_TENSOR_AFFINE = 0,
  PER_CHANNEL_AFFINE = 1,
  PER_TENSOR_SYMMETRIC = 2,
  PER_CHANNEL_SYMMETRIC = 3,
  PER_CHANNEL_AFFINE_FLOAT_QPARAMS = 4,
  COMPILE_TIME_NUM_QSCHEMES = 5,
};

constexpr auto kPerTensorAffine = QScheme::PER_TENSOR_AFFINE;
constexpr auto kPerChannelAffine = QScheme::PER_CHANNEL_AFFINE;
constexpr auto kPerTensorSymmetric = QScheme::PER_TENSOR_SYMMETRIC;
constexpr auto kPerChannelSymmetric = QScheme::PER_CHANNEL_SYMMETRIC;
constexpr auto kPerChannelAffineFloatQParams =
    QScheme::PER_CHANNEL_AFFINE_FLOAT_QPARAMS;
constexpr int COMPILE_TIME_NUM_QSCHEMES =
    static_cast<int>(QScheme::COMPILE_TIME_NUM_QSCHEMES);

inline std::string toString(QScheme qscheme) {
  switch (qscheme) {
    case kPerTensorAffine:
      return "per_tensor_affine";
    case kPerChannelAffine:
      return "per_channel_affine";
    case kPerTensorSymmetric:
      return "per_tensor_symmetric";
    case kPerChannelSymmetric:
      return "per_channel_symmetric";
    case kPerChannelAffineFloatQParams:
      return "per_channel_affine_float_qparams";
    default:
      TORCH_CHECK(false, "Unrecognized qscheme: ", static_cast<int>(qscheme));
  }
}

} // namespace c10

C10_DIAGNOSTIC_POP()
