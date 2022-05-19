
#include <c10/core/SymInt.h>
#include <c10/core/SymbolicIntNode.h>

namespace c10 {

std::shared_ptr<SymbolicIntNode> SymInt::toSymbolicIntNode() {
  auto& st = getSymIntTable();
  TORCH_CHECK(is_symbolic(), "SymInt isn't symbolic");
  return st.getNode(SymInt::SYM_TAG_MASK ^ static_cast<uint64_t>(data_));
}

c10::SymInt SymInt::toSymInt(std::shared_ptr<SymbolicIntNode> sin_sp) {
  auto& sit = getSymIntTable();
  auto index = sit.addNode(sin_sp);
  TORCH_CHECK(
      index != -1,
      "PyTorch doesn't support more than std::numeric_limits<uint64_t>::max() symints");
  auto data = index | SYM_TAG_MASK;
  return c10::SymInt(data);
}
} // namespace c10
