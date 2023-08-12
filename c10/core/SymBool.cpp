#include <c10/core/ConstantSymNodeImpl.h>
#include <c10/core/SymBool.h>
#include <c10/core/SymNodeImpl.h>
#include <array>
#include <utility>

namespace c10 {

SymNode SymBool::promoteToConstantSymNode() const {
  TORCH_CHECK(!is_heap_allocated())
  return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(data_));
}

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

// See SymInt.cpp's DEFINE_BINARY for a small note
#define DEFINE_BINARY(API, OP, METHOD, RET)                              \
  RET SymBool::API(const SymBool& sci) const {                           \
    if (auto ma = maybe_as_bool()) {                                     \
      if (auto mb = sci.maybe_as_bool()) {                               \
        return RET(OP(*ma, *mb));                                        \
      } else {                                                           \
        auto b = sci.toSymNodeImpl();                                    \
        return RET(b->METHOD(b->wrap_bool(*ma)));                        \
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

bool SymBool::has_hint() const {
  if (auto ma = maybe_as_bool()) {
    return true;
  }
  return toSymNodeImpl()->has_hint();
}

} // namespace c10
