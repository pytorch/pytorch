#include <c10/core/SymIntNodeImpl.h>

namespace c10 {

uint64_t SymIntTable::addNode(SymIntNode sin) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto index = nodes_.size();
  nodes_.push_back(sin);
  return index;
}
SymIntNode SymIntTable::getNode(size_t index) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(index < nodes_.size());
  return nodes_[index];
}

c10::SymInt SymIntNodeImpl::toSymInt() {
  // We will need to figure out a way
  // to dedup nodes
  auto sit_sp = this->shared_from_this();
  return SymInt::toSymInt(sit_sp);
}

SymIntTable& getSymIntTable() {
  static SymIntTable sit;
  return sit;
}

} // namespace c10
