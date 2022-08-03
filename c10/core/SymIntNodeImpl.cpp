#include <c10/core/SymInt.h>
#include <c10/core/SymIntNodeImpl.h>

namespace c10 {

c10::SymInt SymIntNodeImpl::toSymInt() {
  auto sit_sp = SymIntNode::reclaim_copy(this);
  return SymInt::toSymInt(sit_sp);
}

} // namespace c10
