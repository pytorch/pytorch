#include <ATen/core/SymIntArrayRef.h>
#include <iostream>

namespace c10 {

at::IntArrayRef expectIntArrayRef(c10::SymIntArrayRef ar) {
  for (c10::SymInt sci : ar) {
    TORCH_CHECK(!sci.is_symbolic());
  }

  return IntArrayRef(reinterpret_cast<const int64_t*>(ar.data()), ar.size());
}

c10::SymIntArrayRef toSymIntArrayRef(at::IntArrayRef ar) {
  return c10::SymIntArrayRef(reinterpret_cast<const c10::SymInt*>(ar.data()), ar.size());
}

std::ostream& operator<<(std::ostream& os, SymInt s) {
  os << "SymInt(" << s.data() << ")";
  return os;
}

std::ostream& operator<<(std::ostream& out, const c10::SymIntArrayRef& list) {
  return out << list.wrapped_symint_array_ref;
}

} // namespace c10
