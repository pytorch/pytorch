#include <c10/core/SymIntArrayRef.h>
#include <iostream>

namespace c10 {

at::IntArrayRef asIntArrayRefSlow(c10::SymIntArrayRef ar) {
  for (c10::SymInt sci : ar) {
    TORCH_CHECK(!sci.is_symbolic());
  }
  return asIntArrayRefUnchecked(ar);
}

at::IntArrayRef asIntArrayRefUnchecked(c10::SymIntArrayRef ar) {
  return IntArrayRef(reinterpret_cast<const int64_t*>(ar.data()), ar.size());
}

std::ostream& operator<<(std::ostream& os, SymInt s) {
  os << "SymInt(" << s.data() << ")";
  return os;
}

} // namespace c10
