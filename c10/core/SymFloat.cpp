#include <c10/core/SymFloat.h>
#include <c10/core/SymFloatNodeImpl.h>
#include <array>

namespace c10 {

SymFloatNode SymFloat::toSymFloatNodeImpl() const {
  TORCH_CHECK(is_symbolic());
  return SymFloatNode::reclaim_copy(toSymFloatNodeImplUnowned());
}

c10::SymFloat SymFloat::toSymFloat(SymFloatNode sin_sp) {
  return c10::SymFloat(std::move(sin_sp));
}

} // namespace c10
