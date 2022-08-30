#include <c10/core/SymFloat.h>
#include <c10/core/SymFloatNodeImpl.h>

namespace c10 {

c10::SymFloat SymFloatNodeImpl::toSymFloat() {
  auto sit_sp = SymFloatNode::reclaim_copy(this);
  return SymFloat::toSymFloat(sit_sp);
}

} // namespace c10
