
#include <c10/core/SymInt.h>
#include <c10/core/SymIntNodeImpl.h>
#include <array>

namespace c10 {

std::array<SymIntNode, 2> normalize_symints(SymInt a_, SymInt b_) {
  SymIntNode a, b;
  if (a_.is_symbolic())
    a = a_.toSymIntNodeImpl();
  if (b_.is_symbolic())
    b = b_.toSymIntNodeImpl();

  SymIntNodeImpl* common = a ? a.get() : b.get();
  // TODO: technically we need to check that the classes match
  if (!a) {
    a = common->wrap(a_.data());
    a_.toSymInt(a); //
  }
  if (!b) {
    b = common->wrap(b_.data());
    b_.toSymInt(b);
  }
  return {a, b};
}

SymIntNode SymInt::toSymIntNodeImpl() const {
  TORCH_CHECK(is_symbolic());
  return SymIntNode::reclaim_copy(toSymIntNodeImplUnowned());
}

c10::SymInt SymInt::toSymInt(SymIntNode sin_sp) {
  auto ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(static_cast<void*>(sin_sp.release())));
  auto rep = (ptr & ~MASK) | IS_SYM;
  return c10::SymInt(static_cast<int64_t>(rep));
}

SymInt SymInt::operator+(SymInt sci) const {
  TORCH_CHECK(
      !this->is_symbolic() && !sci.is_symbolic(),
      "Symbolic Add isn't supported yet");
  return SymInt(data_ + sci.data_);
}

SymInt SymInt::operator*(SymInt sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymInt(data_ * sci.data_);
  }
  auto res = normalize_symints(*this, sci);
  return SymInt::toSymInt(res[0]->mul(res[1]));
}

bool SymInt::operator==(SymInt sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ == sci.data_;
  }
  auto res = normalize_symints(*this, sci);
  return res[0]->eq(res[1])->bool_();
}

bool SymInt::operator!=(SymInt sci) const {
  return !(*this == sci);
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

c10::SymInt SymIntNodeImpl::toSymInt() {
  // We will need to figure out a way
  // to dedup nodes
  auto sit_sp = SymIntNode::reclaim_copy(this);
  return SymInt::toSymInt(sit_sp);
}

} // namespace c10
