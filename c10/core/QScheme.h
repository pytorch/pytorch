#pragma once


namespace c10 {

/**
 * QScheme is an enum that specifies the type of quantization. This has a one
 * to one correspondence with Quantizer
 * Please refer to ATen/core/Quantizer.h to see the Quantizers classes.
 */
enum class QScheme : uint8_t {
  NO_QUANT,
  PER_LAYER_AFFINE,
  PER_CHANNEL_AFFINE,
  PER_LAYER_SYMMETRIC,
  PER_CHANNEL_SYMMETRIC
};

constexpr auto kNoQuant = QScheme::NO_QUANT;
constexpr auto kPerLayerAffine = QScheme::PER_LAYER_AFFINE;

} // namespace c10
