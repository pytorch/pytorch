#include <c10/core/SymIntArrayRef.h>
#include <c10/util/Optional.h>
#include <iostream>

namespace c10 {

at::IntArrayRef asIntArrayRefSlow(c10::SymIntArrayRef ar) {
  auto r = asIntArrayRefSlowOpt(ar);
  TORCH_CHECK(
      r.has_value(),
      "SymIntArrayRef expected to contain only concrete integers");
  return *r;
}

c10::optional<at::IntArrayRef> asIntArrayRefSlowOpt(c10::SymIntArrayRef ar) {
  for (c10::SymInt sci : ar) {
    if (sci.is_symbolic()) {
      return c10::nullopt;
    }
  }

  return {asIntArrayRefUnchecked(ar)};
}

at::IntArrayRef asIntArrayRefUnchecked(c10::SymIntArrayRef ar) {
  return IntArrayRef(reinterpret_cast<const int64_t*>(ar.data()), ar.size());
}

// TODO: this print is bad
std::ostream& operator<<(std::ostream& os, SymInt s) {
  os << "SymInt(" << s.as_int_unchecked() << ")";
  return os;
}

} // namespace c10
