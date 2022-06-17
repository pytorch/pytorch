
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

SymInt SymInt::operator+(SymInt sci) const {
  TORCH_CHECK(
      !this->is_symbolic() && !sci.is_symbolic(),
      "Symbolic Add isn't supported yet");
  return SymInt(data_ + sci.data_);
}

bool SymInt::operator<(SymInt sci) const {
  TORCH_CHECK(
      !this->is_symbolic() && !sci.is_symbolic(),
      "Symbolic lt isn't supported yet");
  return data_ < sci.data_;
}

void SymInt::operator*=(SymInt sci) {
  TORCH_CHECK(
      !this->is_symbolic() && !sci.is_symbolic(),
      "Symbolic mul_ isn't supported yet");
  data_ = data_ * sci.data_;
}

bool SymInt::operator<(int64_t sci) const {
  TORCH_CHECK(!this->is_symbolic(), "Symbolic lt isn't supported yet");
  return data_ < sci;
}

bool SymInt::operator==(int64_t sci) const {
  TORCH_CHECK(!this->is_symbolic(), "Symbolic eq isn't supported yet");
  return data_ == sci;
}

bool SymInt::operator!=(int64_t sci) const {
  TORCH_CHECK(!this->is_symbolic(), "Symbolic neq isn't supported yet");
  return data_ != sci;
}

SymInt SymInt::operator*(int64_t sci) const {
  TORCH_CHECK(!this->is_symbolic(), "Symbolic mul isn't supported yet");
  return SymInt(data_ * sci);
}

} // namespace c10
