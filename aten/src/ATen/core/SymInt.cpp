
#include <ATen/core/SymInt.h>
#include <ATen/core/SymbolicIntNode.h>

namespace c10 {

std::shared_ptr<SymbolicIntNode> SymInt::toSymbolicIntNode() {
  auto& st = getSymIntTable();
  TORCH_CHECK(is_symbolic());
  return st.getNode(SymInt::SYM_TAG_MASK ^ static_cast<uint64_t>(data_));
}

c10::SymInt SymInt::toSymInt(std::shared_ptr<SymbolicIntNode> sin_sp) {
  auto& sit = getSymIntTable();
  auto data = sit.addNode(sin_sp) | SYM_TAG_MASK;
  return c10::SymInt(data);
}
} // namespace c10
