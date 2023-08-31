#include <c10/core/ConstantSymNodeImpl.h>

namespace c10 {

// This is used to support the case where the lhs is a constant symnode
// and the rhs is a singleton symnode. This situation occurs today when we
// perform a binary op between singleton int and plain int and the
// singleton promotes the int into a constant symnode. If we'd like to
// support more combinations in the future, we may need to implement some
// kind of multiple dispatch.
#define DEFINE_BINARY_OP(OP)                                           \
  template <typename T>                                                \
  c10::SymNode ConstantSymNodeImpl<T>::OP(const c10::SymNode& other) { \
    TORCH_INTERNAL_ASSERT(other->singleton_int().has_value());         \
    c10::raw::intrusive_ptr::incref(this);                             \
    return other->OP(                                                  \
        c10::intrusive_ptr<ConstantSymNodeImpl<T>>::reclaim(this));    \
  }

DEFINE_BINARY_OP(eq)
DEFINE_BINARY_OP(ne)
DEFINE_BINARY_OP(ge)
DEFINE_BINARY_OP(lt)

#undef DEFINE_BINARY_OP

template class ConstantSymNodeImpl<bool>;
template class ConstantSymNodeImpl<int64_t>;

} // namespace c10
