#include <c10/core/ConstantSymNodeImpl.h>

namespace c10 {

// This is used to support the case where the lhs is a constant symnode
// and the rhs is a singleton symnode. This situation occurs today when we
// perform a binary op between singleton int and plain int and the
// singleton promotes the int into a constant symnode. If we'd like to
// support more combinations in the future, we may need to implement some
// kind of multiple dispatch.
#define DEFINE_BINARY_OP(OP, ROP)                                        \
  template <typename T>                                                  \
  c10::SymNode ConstantSymNodeImpl<T>::OP(const c10::SymNode& other) {   \
    TORCH_INTERNAL_ASSERT(other->singleton_int().has_value());           \
    return other->ROP(                                                   \
        c10::intrusive_ptr<ConstantSymNodeImpl<T>>::reclaim_copy(this)); \
  }

DEFINE_BINARY_OP(eq, eq)
DEFINE_BINARY_OP(ne, ne)
DEFINE_BINARY_OP(ge, le)
DEFINE_BINARY_OP(le, ge)
DEFINE_BINARY_OP(lt, gt)
DEFINE_BINARY_OP(gt, lt)

#undef DEFINE_BINARY_OP

template class ConstantSymNodeImpl<bool>;
template class ConstantSymNodeImpl<int64_t>;

} // namespace c10
