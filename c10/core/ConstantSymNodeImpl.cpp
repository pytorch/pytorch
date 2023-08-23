#include <c10/core/ConstantSymNodeImpl.h>

namespace c10 {

#define DEFINE_BINARY_OP(OP) \
  template <typename T> \
  c10::SymNode ConstantSymNodeImpl<T>::OP(const c10::SymNode& other) { \
    TORCH_INTERNAL_ASSERT(other->singleton_int().has_value()); \
    c10::raw::intrusive_ptr::incref(this); \
    return other->OP(c10::intrusive_ptr<ConstantSymNodeImpl<T>>::reclaim(this)); \
  }

DEFINE_BINARY_OP(eq)
DEFINE_BINARY_OP(ne)
DEFINE_BINARY_OP(ge)
DEFINE_BINARY_OP(lt)

#undef DEFINE_BINARY_OP

template class ConstantSymNodeImpl<bool>;
template class ConstantSymNodeImpl<int64_t>;

} // namespace c10
