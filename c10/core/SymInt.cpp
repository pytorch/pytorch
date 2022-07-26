
#include <c10/core/SymInt.h>
#include <c10/core/SymbolicIntNode.h>

namespace c10 {

std::shared_ptr<SymbolicIntNode> SymInt::toSymbolicIntNode() const {
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

bool SymInt::operator!=(SymInt sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ != sci.data_;
  }
  // TODO: This is way to much boilerplate
  std::shared_ptr<SymbolicIntNode> a =
      is_symbolic() ? toSymbolicIntNode() : nullptr;
  std::shared_ptr<SymbolicIntNode> b =
      sci.is_symbolic() ? sci.toSymbolicIntNode() : nullptr;

  SymbolicIntNode* common = a ? a.get() : b.get();
  // TODO: technically we need to check that the classes match
  if (!a) {
    a = common->wrap(data_);
    toSymInt(a); //
  }
  if (!b) {
    b = common->wrap(sci.data_);
    toSymInt(b);
  }

  auto c = a->ne(b);
  return c->bool_();
}

bool SymInt::operator==(SymInt sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ == sci.data_;
  }
  // TODO: This is way to much boilerplate
  std::shared_ptr<SymbolicIntNode> a =
      is_symbolic() ? toSymbolicIntNode() : nullptr;
  std::shared_ptr<SymbolicIntNode> b =
      sci.is_symbolic() ? sci.toSymbolicIntNode() : nullptr;

  SymbolicIntNode* common = a ? a.get() : b.get();
  // TODO: technically we need to check that the classes match
  if (!a) {
    a = common->wrap(data_);
    toSymInt(a); //
  }
  if (!b) {
    b = common->wrap(sci.data_);
    toSymInt(b);
  }

  auto c = a->eq(b);
  return c->bool_();
}

SymInt SymInt::operator*(SymInt sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymInt(data_ * sci.data_);
  }
  // TODO: This is way to much boilerplate
  std::shared_ptr<SymbolicIntNode> a =
      is_symbolic() ? toSymbolicIntNode() : nullptr;
  std::shared_ptr<SymbolicIntNode> b =
      sci.is_symbolic() ? sci.toSymbolicIntNode() : nullptr;

  SymbolicIntNode* common = a ? a.get() : b.get();
  // TODO: technically we need to check that the classes match
  if (!a) {
    a = common->wrap(data_);
    toSymInt(a); //
  }
  if (!b) {
    b = common->wrap(sci.data_);
    toSymInt(b);
  }
  return SymInt::toSymInt(a->add(b));
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
  return *this == c10::SymInt(sci);
}

bool SymInt::operator!=(int64_t sci) const {
  return *this != c10::SymInt(sci);
}

SymInt SymInt::operator*(int64_t sci) const {
  TORCH_CHECK(!this->is_symbolic(), "Symbolic mul isn't supported yet");
  return SymInt(data_ * sci);
}

} // namespace c10
