#include <c10/core/SymBool.h>
#include <c10/core/SymNodeImpl.h>
#include <array>
#include <utility>

namespace c10 {

SymNode SymBool::toSymNodeImpl() const {
  TORCH_CHECK(is_symbolic());
  return SymNode::reclaim_copy(toSymNodeImplUnowned());
}

SymNode SymBool::wrap_node(const SymNode& base) const {
  if (is_symbolic()) {
    return toSymNodeImpl();
  } else {
    return base->wrap_bool(as_bool_unchecked());
  }
}

static std::array<SymNode, 2> normalize_symbools(
    const SymBool& a_,
    const SymBool& b_) {
  SymNode a, b;
  if (a_.is_symbolic())
    a = a_.toSymNodeImpl();
  if (b_.is_symbolic())
    b = b_.toSymNodeImpl();

  SymNodeImpl* common = a ? a.get() : b.get();
  if (!a) {
    a = common->wrap_bool(a_.as_bool_unchecked());
  }
  if (!b) {
    b = common->wrap_bool(b_.as_bool_unchecked());
  }
  return {std::move(a), std::move(b)};
}

SymBool SymBool::sym_and(const SymBool& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymBool(data_ && sci.data_);
  }
  auto res = normalize_symbools(*this, sci);
  return SymBool(res[0]->sym_and(res[1]));
}

SymBool SymBool::sym_or(const SymBool& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymBool(data_ || sci.data_);
  }
  auto res = normalize_symbools(*this, sci);
  return SymBool(res[0]->sym_or(res[1]));
}

SymBool SymBool::sym_not() const {
  if (!is_symbolic()) {
    return SymBool(!data_);
  }
  return SymBool(toSymNodeImpl()->sym_not());
}

std::ostream& operator<<(std::ostream& os, const SymBool& s) {
  if (s.is_symbolic()) {
    os << s.toSymNodeImpl()->str();
  } else {
    os << s.as_bool_unchecked();
  }
  return os;
}

bool SymBool::guard_bool(const char* file, int64_t line) const {
  if (!is_symbolic()) {
    return data_;
  }
  SymNode a = toSymNodeImpl();
  return a->guard_bool(file, line);
}

bool SymBool::expect_true(const char* file, int64_t line) const {
  if (!is_symbolic()) {
    return data_;
  }
  SymNode a = toSymNodeImpl();
  return a->expect_true(file, line);
}

bool SymBool::has_hint() const {
  if (!is_symbolic()) {
    return true;
  }
  return toSymNodeImpl()->has_hint();
}

} // namespace c10
