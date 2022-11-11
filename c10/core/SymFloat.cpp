#include <c10/core/SymFloat.h>
#include <c10/core/SymNodeImpl.h>
#include <array>
#include <utility>

namespace c10 {

SymNode SymFloat::toSymNodeImpl() const {
  TORCH_CHECK(is_symbolic());
  return SymNode::reclaim_copy(toSymNodeImplUnowned());
}

static std::array<SymNode, 2> normalize_symfloats(
    const SymFloat& a_,
    const SymFloat& b_) {
  SymNode a, b;
  if (a_.is_symbolic())
    a = a_.toSymNodeImpl();
  if (b_.is_symbolic())
    b = b_.toSymNodeImpl();

  SymNodeImpl* common = a ? a.get() : b.get();
  if (!a) {
    a = common->wrap_float(a_.as_float_unchecked());
  }
  if (!b) {
    b = common->wrap_float(b_.as_float_unchecked());
  }
  return {std::move(a), std::move(b)};
}

SymFloat SymFloat::operator+(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymFloat(data_ + sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat(res[0]->add(res[1]));
}

SymFloat SymFloat::operator-(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymFloat(data_ - sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat(res[0]->sub(res[1]));
}

SymFloat SymFloat::operator*(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymFloat(data_ * sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat(res[0]->mul(res[1]));
}

SymFloat SymFloat::operator/(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymFloat(data_ / sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat(res[0]->truediv(res[1]));
}

std::ostream& operator<<(std::ostream& os, const SymFloat& s) {
  if (s.is_symbolic()) {
    os << s.toSymNodeImpl()->str();
  } else {
    os << s.as_float_unchecked();
  }
  return os;
}

double SymFloat::guard_float(const char* file, int64_t line) const {
  if (!is_symbolic()) {
    return data_;
  }
  SymNode a = toSymNodeImpl();
  return a->guard_float(file, line);
}

} // namespace c10
