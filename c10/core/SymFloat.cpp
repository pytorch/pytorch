#include <c10/core/SymFloat.h>
#include <c10/core/SymFloatNodeImpl.h>
#include <array>

namespace c10 {

SymFloatNode SymFloat::toSymFloatNodeImpl() const {
  TORCH_CHECK(is_symbolic());
  return SymFloatNode::reclaim_copy(toSymFloatNodeImplUnowned());
}

static std::array<SymFloatNode, 2> normalize_symfloats(
    SymFloat a_,
    SymFloat b_) {
  SymFloatNode a, b;
  if (a_.is_symbolic())
    a = a_.toSymFloatNodeImpl();
  if (b_.is_symbolic())
    b = b_.toSymFloatNodeImpl();

  SymFloatNodeImpl* common = a ? a.get() : b.get();
  // TODO: technically we need to check that the classes match
  if (!a) {
    a = common->wrap(a_.as_float_unchecked());
    a_.toSymFloat(a); //
  }
  if (!b) {
    b = common->wrap(b_.as_float_unchecked());
    b_.toSymFloat(b);
  }
  return {a, b};
}

SymFloat SymFloat::operator+(SymFloat sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymFloat(data_ + sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat::toSymFloat(res[0]->add(res[1]));
}

SymFloat SymFloat::operator-(SymFloat sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymFloat(data_ - sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat::toSymFloat(res[0]->sub(res[1]));
}

SymFloat SymFloat::operator*(SymFloat sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymFloat(data_ * sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat::toSymFloat(res[0]->mul(res[1]));
}

SymFloat SymFloat::operator/(SymFloat sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymFloat(data_ / sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat::toSymFloat(res[0]->truediv(res[1]));
}

c10::SymFloat SymFloat::toSymFloat(SymFloatNode sin_sp) {
  return c10::SymFloat(std::move(sin_sp));
}

std::ostream& operator<<(std::ostream& os, SymFloat s) {
  if (s.is_symbolic()) {
    os << s.toSymFloatNodeImpl()->str();
  } else {
    os << s.as_float_unchecked();
  }
  return os;
}

} // namespace c10
