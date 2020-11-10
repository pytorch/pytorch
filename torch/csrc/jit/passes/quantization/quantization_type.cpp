#include <torch/csrc/jit/passes/quantization/quantization_type.h>

namespace torch {
namespace jit {

std::ostream& operator<<(std::ostream& os, QuantType t) {
  switch (t) {
    case QuantType::DYNAMIC:
      os << "dynamic";
      break;
    case QuantType::STATIC:
      os << "static";
      break;
    case QuantType::QAT:
      os << "qat";
      break;
    case QuantType::WEIGHT_ONLY:
      os << "weight_only";
      break;
    case QuantType::ACTIVATION_ONLY:
      os << "activation_only";
      break;
    default:
      os.setstate(std::ios_base::failbit);
  }
  return os;
}

} // namespace jit
} // namespace torch
