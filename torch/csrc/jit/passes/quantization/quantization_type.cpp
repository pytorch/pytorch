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
    default:
      os.setstate(std::ios_base::failbit);
  }
  return os;
}

} // namespace jit
} // namespace torch
