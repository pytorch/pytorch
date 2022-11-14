#include <c10/core/SymFloat.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymNodeImpl.h>
#include <array>
#include <utility>

namespace c10 {

static std::array<SymNode, 2> normalize_symints(
    const SymInt& a_,
    const SymInt& b_) {
  SymNode a, b;
  if (a_.is_symbolic())
    a = a_.toSymNodeImpl();
  if (b_.is_symbolic())
    b = b_.toSymNodeImpl();

  SymNodeImpl* common = a ? a.get() : b.get();
  // TODO: technically we need to check that the classes match
  if (!a) {
    a = common->wrap_int(a_.as_int_unchecked());
  }
  if (!b) {
    b = common->wrap_int(b_.as_int_unchecked());
  }
  return {std::move(a), std::move(b)};
}

SymNode SymInt::toSymNodeImpl() const {
  TORCH_CHECK(is_symbolic());
  return SymNode::reclaim_copy(toSymNodeImplUnowned());
}

SymInt::SymInt(SymNode sin_sp) {
  TORCH_CHECK(sin_sp->is_int());
  auto ptr = static_cast<uint64_t>(
      reinterpret_cast<uintptr_t>(static_cast<void*>(sin_sp.release())));
  auto rep = (ptr & ~MASK) | IS_SYM;
  data_ = static_cast<int64_t>(rep);
}

int64_t SymInt::guard_int(const char* file, int64_t line) const {
  if (!is_symbolic()) {
    return data_;
  }
  SymNode a = toSymNodeImpl();
  return a->guard_int(file, line);
}

SymInt::operator SymFloat() const {
  if (!is_symbolic()) {
    return SymFloat(double(data_));
  }
  return SymFloat(toSymNodeImpl()->sym_float());
}

SymInt SymInt::operator+(const SymInt& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymInt(data_ + sci.data_);
  }
  auto res = normalize_symints(*this, sci);
  return SymInt(res[0]->add(res[1]));
}

SymInt SymInt::operator-(const SymInt& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymInt(data_ - sci.data_);
  }
  auto res = normalize_symints(*this, sci);
  return SymInt(res[0]->sub(res[1]));
}

SymInt SymInt::operator*(const SymInt& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymInt(data_ * sci.data_);
  }
  auto res = normalize_symints(*this, sci);
  return SymInt(res[0]->mul(res[1]));
}

SymInt SymInt::operator/(const SymInt& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymInt(data_ / sci.data_);
  }
  auto res = normalize_symints(*this, sci);
  return SymInt(res[0]->floordiv(res[1]));
}

SymInt SymInt::operator%(const SymInt& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymInt(data_ % sci.data_);
  }
  auto res = normalize_symints(*this, sci);
  return SymInt(res[0]->mod(res[1]));
}

bool SymInt::operator==(const SymInt& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ == sci.data_;
  }
  auto res = normalize_symints(*this, sci);
  return res[0]->eq(res[1])->bool_();
}

bool SymInt::operator!=(const SymInt& sci) const {
  return !(*this == sci);
}

bool SymInt::operator<(const SymInt& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ < sci.data_;
  }
  auto res = normalize_symints(*this, sci);
  return res[0]->lt(res[1])->bool_();
}

bool SymInt::operator<=(const SymInt& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ <= sci.data_;
  }
  auto res = normalize_symints(*this, sci);
  return res[0]->le(res[1])->bool_();
}

bool SymInt::operator>(const SymInt& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ > sci.data_;
  }
  auto res = normalize_symints(*this, sci);
  return res[0]->gt(res[1])->bool_();
}

bool SymInt::operator>=(const SymInt& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ >= sci.data_;
  }
  auto res = normalize_symints(*this, sci);
  return res[0]->ge(res[1])->bool_();
}

SymInt SymInt::min(const SymInt& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return std::min(data_, sci.data_);
  }
  auto res = normalize_symints(*this, sci);
  return SymInt(res[0]->min(res[1]));
}
SymInt SymInt::max(const SymInt& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return std::max(data_, sci.data_);
  }
  auto res = normalize_symints(*this, sci);
  return SymInt(res[0]->max(res[1]));
}

void SymInt::operator*=(const SymInt& sci) {
  *this = *this * sci;
}

void SymInt::operator/=(const SymInt& sci) {
  *this = *this / sci;
}

void SymInt::operator+=(const SymInt& sci) {
  *this = *this + sci;
}

bool SymInt::operator<(int64_t sci) const {
  return *this < c10::SymInt(sci);
}

bool SymInt::operator<=(int64_t sci) const {
  return *this <= c10::SymInt(sci);
}

bool SymInt::operator>(int64_t sci) const {
  return *this > c10::SymInt(sci);
}

bool SymInt::operator>=(int64_t sci) const {
  return *this >= c10::SymInt(sci);
}

bool SymInt::operator==(int64_t sci) const {
  return *this == c10::SymInt(sci);
}

bool SymInt::operator!=(int64_t sci) const {
  return *this != c10::SymInt(sci);
}

SymInt SymInt::operator*(int64_t sci) const {
  return *this * c10::SymInt(sci);
}

std::ostream& operator<<(std::ostream& os, const SymInt& s) {
  if (s.is_symbolic()) {
    os << s.toSymNodeImpl()->str();
  } else {
    os << s.as_int_unchecked();
  }
  return os;
}

SymInt operator-(const SymInt& s) {
  if (s.is_symbolic()) {
    return SymInt(s.toSymNodeImpl()->neg());
  } else {
    return SymInt(-s.as_int_unchecked());
  }
}

} // namespace c10
