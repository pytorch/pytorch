#include <c10/core/SymFloat.h>
#include <c10/core/SymNodeImpl.h>
#include <array>
#include <cmath>
#include <utility>

namespace c10 {

SymNode SymFloat::toSymNodeImpl() const {
  TORCH_CHECK(is_symbolic());
  return SymNode::reclaim_copy(toSymNodeImplUnowned());
}

SymNode SymFloat::wrap_node(const SymNode& base) const {
  if (is_symbolic()) {
    return toSymNodeImpl();
  } else {
    return base->wrap_float(as_float_unchecked());
  }
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

SymBool SymFloat::sym_eq(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ == sci.data_;
  }
  auto res = normalize_symfloats(*this, sci);
  return res[0]->eq(res[1]);
}

SymBool SymFloat::sym_ne(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ != sci.data_;
  }
  auto res = normalize_symfloats(*this, sci);
  return res[0]->ne(res[1]);
}

SymBool SymFloat::sym_lt(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ < sci.data_;
  }
  auto res = normalize_symfloats(*this, sci);
  return res[0]->lt(res[1]);
}

SymBool SymFloat::sym_le(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ <= sci.data_;
  }
  auto res = normalize_symfloats(*this, sci);
  return res[0]->le(res[1]);
}

SymBool SymFloat::sym_gt(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ > sci.data_;
  }
  auto res = normalize_symfloats(*this, sci);
  return res[0]->gt(res[1]);
}

SymBool SymFloat::sym_ge(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ >= sci.data_;
  }
  auto res = normalize_symfloats(*this, sci);
  return res[0]->ge(res[1]);
}

SymFloat SymFloat::min(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return std::min(data_, sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat(res[0]->sym_min(res[1]));
}
SymFloat SymFloat::max(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return std::max(data_, sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat(res[0]->sym_max(res[1]));
}

std::ostream& operator<<(std::ostream& os, const SymFloat& s) {
  if (s.is_symbolic()) {
    os << s.toSymNodeImpl()->str();
  } else {
    os << s.as_float_unchecked();
  }
  return os;
}

SymFloat SymFloat::sqrt() const {
  if (!is_symbolic()) {
    return SymFloat(std::sqrt(data_));
  }
  auto other = SymFloat(-0.5);
  auto res = normalize_symfloats(*this, other);
  return SymFloat(res[0]->pow(res[1]));
}

double SymFloat::guard_float(const char* file, int64_t line) const {
  if (!is_symbolic()) {
    return data_;
  }
  SymNode a = toSymNodeImpl();
  return a->guard_float(file, line);
}

bool SymFloat::has_hint() const {
  if (!is_symbolic()) {
    return true;
  }
  return toSymNodeImpl()->has_hint();
}

} // namespace c10
