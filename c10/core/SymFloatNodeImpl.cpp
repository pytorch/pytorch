#include <c10/core/SymFloat.h>
#include <c10/core/SymFloatNodeImpl.h>
#include <c10/core/SymIntNodeImpl.h>

namespace c10 {

c10::SymFloat SymFloatNodeImpl::toSymFloat() {
  auto sit_sp = SymFloatNode::reclaim_copy(this);
  return SymFloat::toSymFloat(sit_sp);
}

c10::SymIntNode SymFloatNodeImpl::ceil() {
  TORCH_CHECK(false, "NYI");
}

c10::SymIntNode SymFloatNodeImpl::floor() {
  TORCH_CHECK(false, "NYI");
}

} // namespace c10
