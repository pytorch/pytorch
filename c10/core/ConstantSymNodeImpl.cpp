#include <c10/core/ConstantSymNodeImpl.h>

namespace c10 {

// This is used to support the case where the lhs is a constant symnode
// and the rhs is a nested int symnode. This situation occurs today when we
// perform a binary op between nested int and plain int and the
// int is promoted into a constant symnode. If we'd like to
// support more combinations in the future, we may need to implement some
// kind of multiple dispatch.
#define DEFINE_BINARY_OP(OP, ROP)                                        \
  template <typename T>                                                  \
  c10::SymNode ConstantSymNodeImpl<T>::OP(const c10::SymNode& other) {   \
    TORCH_INTERNAL_ASSERT(other->is_nested_int());                       \
    return other->ROP(                                                   \
        c10::intrusive_ptr<ConstantSymNodeImpl<T>>::reclaim_copy(this)); \
  }

DEFINE_BINARY_OP(eq, eq)
DEFINE_BINARY_OP(ne, ne)
DEFINE_BINARY_OP(ge, le)
DEFINE_BINARY_OP(le, ge)
DEFINE_BINARY_OP(lt, gt)
DEFINE_BINARY_OP(gt, lt)
DEFINE_BINARY_OP(mul, mul)

#undef DEFINE_BINARY_OP

template <typename T>
c10::SymNode ConstantSymNodeImpl<T>::sym_and(const c10::SymNode& other) {
  TORCH_INTERNAL_ASSERT(is_bool_(), "sym_and only works on bool");

  // If other is also a constant bool, compute the result directly
  if (auto other_const = other->constant_bool()) {
    bool result = this->bool_() && *other_const;
    return c10::make_intrusive<ConstantSymNodeImpl<bool>>(result);
  }

  // If other is a nested int or other symbolic type, defer to it
  return other->sym_and(
      c10::intrusive_ptr<ConstantSymNodeImpl<T>>::reclaim_copy(this));
}

template <typename T>
c10::SymNode ConstantSymNodeImpl<T>::sym_or(const c10::SymNode& other) {
  TORCH_INTERNAL_ASSERT(is_bool_(), "sym_or only works on bool");

  // If other is also a constant bool, compute the result directly
  if (auto other_const = other->constant_bool()) {
    bool result = this->bool_() || *other_const;
    return c10::make_intrusive<ConstantSymNodeImpl<bool>>(result);
  }

  // If other is a nested int or other symbolic type, defer to it
  return other->sym_or(
      c10::intrusive_ptr<ConstantSymNodeImpl<T>>::reclaim_copy(this));
}

template class ConstantSymNodeImpl<bool>;
template class ConstantSymNodeImpl<int64_t>;

} // namespace c10
