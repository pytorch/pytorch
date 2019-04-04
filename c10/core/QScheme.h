#pragma once

namespace c10 {

/**
 * QScheme is an enum that specifies the type of quantization. This has a one
 * to one correspondence with Quantizer
 * Please refer to ATen/core/Quantizer.h to see the Quantizers classes.
 */
enum class QScheme : uint8_t {
  NO_QUANT,
  PER_TENSOR_AFFINE,
  PER_CHANNEL_AFFINE,
  PER_TENSOR_SYMMETRIC,
  PER_CHANNEL_SYMMETRIC
};

constexpr auto kNoQuant = QScheme::NO_QUANT;
constexpr auto kPerTensorAffine = QScheme::PER_TENSOR_AFFINE;
constexpr auto kPerChannelAffine = QScheme::PER_CHANNEL_AFFINE;
constexpr auto kPerTensorSymmetric = QScheme::PER_TENSOR_SYMMETRIC;
constexpr auto kPerChannelSymmetric = QScheme::PER_CHANNEL_SYMMETRIC;

} // namespace c10
