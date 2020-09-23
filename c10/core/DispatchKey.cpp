#include <c10/core/DispatchKey.h>

namespace c10 {
const char* toString(DispatchKey t) {
#define DEFINE_CASE(dk) \
  case DispatchKey::dk: \
    return #dk;

  switch (t) {
    FOR_EACH_RUNTIME_DISPATCH_KEY(DEFINE_CASE)
    FOR_EACH_ALIAS_DISPATCH_KEY(DEFINE_CASE)
    default:
      return "UNKNOWN_TENSOR_TYPE_ID";
  }
#undef DEFINE_CASE
}

std::ostream& operator<<(std::ostream& str, DispatchKey rhs) {
  return str << toString(rhs);
}

DispatchKey getAutogradKeyFromBackend(DispatchKey t) {
#define DEFINE_CASE(dk) \
  case DispatchKey::dk: \
    return DispatchKey::Autograd##dk;

  switch (t) {
    FOR_EACH_BACKEND_DISPATCH_KEY_WITH_AUTOGRAD_KEY(DEFINE_CASE)
    default:
      return DispatchKey::AutogradOther;
  }
#undef DEFINE_CASE
}

} // namespace c10
