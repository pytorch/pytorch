#include <c10/core/SymIntNodeImpl.h>
#include <c10/core/SymInt.h>

namespace c10 {

c10::SymInt SymIntNodeImpl::toSymInt() {
  // We will need to figure out a way
  // to dedup nodes
  auto sit_sp = SymIntNode::reclaim_copy(this);
  return SymInt::toSymInt(sit_sp);
}

} // namespace c10
