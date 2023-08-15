#include <c10/core/ConstantSymNodeImpl.h>

namespace c10 {
// Temporary hack to avoid having to implement multiple dispatch for now
// Currently even if we have this method, we still raise an error when we get
// to SingletonSymNode::eq since comparing with non-singleton is disallowed.
// However, we may change that behavior in the future.
template <typename T>
c10::SymNode ConstantSymNodeImpl<T>::eq(const c10::SymNode& other) {
  TORCH_INTERNAL_ASSERT(other->singleton_int().has_value());
  c10::raw::intrusive_ptr::incref(this);
  return other->eq(c10::intrusive_ptr<ConstantSymNodeImpl<T>>::reclaim(this));
}
template <typename T>
c10::SymNode ConstantSymNodeImpl<T>::ne(const c10::SymNode& other) {
  TORCH_INTERNAL_ASSERT(other->singleton_int().has_value());
  c10::raw::intrusive_ptr::incref(this);
  return other->ne(c10::intrusive_ptr<ConstantSymNodeImpl<T>>::reclaim(this));
}

template class ConstantSymNodeImpl<bool>;
template class ConstantSymNodeImpl<int64_t>;
} // namespace c10
