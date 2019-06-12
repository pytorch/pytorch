#pragma once

#include <c10/core/TensorTypeId.h>
#include <c10/core/DeviceType.h>
// For kCPU
// TODO: Why are these defined in Backend.h?
#include <c10/core/Backend.h>

namespace c10 {

/**
 * QScheme is an enum that specifies the type of quantization. This has a one
 * to one correspondence with Quantizer
 * Please refer to ATen/core/Quantizer.h to see the Quantizers classes.
 * Keep this file in sync with torch/nn/_qscheme.py
 */
enum class QScheme : uint8_t {
  PER_TENSOR_AFFINE = 0,
  PER_CHANNEL_AFFINE = 1,
  PER_TENSOR_SYMMETRIC = 2,
  PER_CHANNEL_SYMMETRIC = 3,
  COMPILE_TIME_NUM_QSCHEMES = 4,
};

constexpr auto kPerTensorAffine = QScheme::PER_TENSOR_AFFINE;
constexpr auto kPerChannelAffine = QScheme::PER_CHANNEL_AFFINE;
constexpr auto kPerTensorSymmetric = QScheme::PER_TENSOR_SYMMETRIC;
constexpr auto kPerChannelSymmetric = QScheme::PER_CHANNEL_SYMMETRIC;
constexpr int COMPILE_TIME_NUM_QSCHEMES =
  static_cast<int>(QScheme::COMPILE_TIME_NUM_QSCHEMES);

inline std::string toString(QScheme qscheme) {
  switch(qscheme) {
    case kPerTensorAffine:
      return "PerTensorAffine";
    case kPerChannelAffine:
      return "PerChannelAffine";
    case kPerTensorSymmetric:
      return "PerTensorSymmetric";
    case kPerChannelSymmetric:
      return "PerChannelSymmetric";
    default:
      AT_ERROR("Unrecognized qscheme: ", static_cast<int>(qscheme));
  }
}

} // namespace c10
