#pragma once

/*
namespace c10 {
// QType is an enum that specifies the type of quantization
enum class QType : int8_t {
  AFFINE,
  SYMMETRIC
};

// QuantScheme is a specification for quantization
struct C10_API QScheme {
  QScheme(QType qtype = QType::AFFINE, int8_t num_bits = 8, bool is_per_layer = true) : qtype_(qtype), num_bits_(num_bits), is_per_layer_(is_per_layer) {}

  QType qtype_{QType::AFFINE}; // 8-bit
  int8_t num_bits_{8}; // 8-bit
  // specifies whether the quantization is applied to
  // the whole tensor
  bool is_per_layer{true}; // 16-bit because of alignment?
};

constexpr auto kPerLayerAffine8Bit = QScheme();

} // namespace c10
*/
