
#include <c10/core/SymInt.h>
#include <c10/core/SymbolicIntNode.h>

namespace c10 {

std::shared_ptr<SymbolicIntNode> SymInt::toSymbolicIntNode() {
  auto& st = getSymIntTable();
  TORCH_CHECK(is_symbolic());
  return st.getNode(static_cast<uint64_t>(data_) & ~MASK);
}

c10::SymInt SymInt::toSymInt(std::shared_ptr<SymbolicIntNode> sin_sp) {
  auto& sit = getSymIntTable();
  uint64_t idx = sit.addNode(sin_sp);
  TORCH_CHECK(idx < MAX_SYM_IDX, "SymbolicIntNode index overflow: ", idx);
  uint64_t data = idx | IS_SYM;
  return c10::SymInt(static_cast<int64_t>(data));
}
} // namespace c10
