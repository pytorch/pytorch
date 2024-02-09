#include <c10/core/SymBool.h>
#include <c10/core/SymNodeImpl.h>

namespace c10 {

SymNode SymBool::toSymNodeImpl() const {
  TORCH_CHECK(is_heap_allocated());
  return SymNode::reclaim_copy(toSymNodeImplUnowned());
}

SymNode SymBool::wrap_node(const SymNode& base) const {
  if (auto ma = maybe_as_bool()) {
    return base->wrap_bool(*ma);
  } else {
    return toSymNodeImpl();
  }
}

#define DEFINE_BINARY(API, OP, METHOD, RET)                              \
  RET SymBool::API(const SymBool& sci) const {                           \
    if (auto ma = maybe_as_bool()) {                                     \
      if (auto mb = sci.maybe_as_bool()) {                               \
        return RET(OP(*ma, *mb));                                        \
      } else {                                                           \
        auto b = sci.toSymNodeImpl();                                    \
        return RET(b->wrap_bool(*ma)->METHOD(b));                        \
      }                                                                  \
    } else {                                                             \
      if (auto mb = sci.maybe_as_bool()) {                               \
        auto a = toSymNodeImplUnowned();                                 \
        return RET(a->METHOD(a->wrap_bool(*mb)));                        \
      } else {                                                           \
        return RET(toSymNodeImplUnowned()->METHOD(sci.toSymNodeImpl())); \
      }                                                                  \
    }                                                                    \
  }

// clang-format off
DEFINE_BINARY(sym_and, std::logical_and<>(), sym_and, SymBool)
DEFINE_BINARY(sym_or, std::logical_or<>(), sym_or, SymBool)
// clang-format on

SymBool SymBool::sym_not() const {
  if (auto ma = maybe_as_bool()) {
    return SymBool(!*ma);
  }
  return SymBool(toSymNodeImpl()->sym_not());
}

std::ostream& operator<<(std::ostream& os, const SymBool& s) {
  if (auto ma = s.maybe_as_bool()) {
    os << *ma;
  } else {
    os << s.toSymNodeImpl()->str();
  }
  return os;
}

bool SymBool::guard_bool(const char* file, int64_t line) const {
  if (auto ma = maybe_as_bool()) {
    return *ma;
  }
  SymNode a = toSymNodeImpl();
  return a->guard_bool(file, line);
}

bool SymBool::guard_size_oblivious(const char* file, int64_t line) const {
  if (auto ma = maybe_as_bool()) {
    return *ma;
  }
  SymNode a = toSymNodeImpl();
  return a->guard_size_oblivious(file, line);
}

bool SymBool::expect_true(const char* file, int64_t line) const {
  if (auto ma = maybe_as_bool()) {
    return *ma;
  }
  SymNode a = toSymNodeImpl();
  return a->expect_true(file, line);
}

bool SymBool::has_hint() const {
  if (auto ma = maybe_as_bool()) {
    return true;
  }
  return toSymNodeImpl()->has_hint();
}

} // namespace c10
