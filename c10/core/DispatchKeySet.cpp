#include <c10/core/DispatchKeySet.h>

namespace c10 {
// math_dispatch_keyset contains all keys in backend_dispatch_keyset and autograd_dispatch_keyset
// Alias key DispatchKey::Math maps to math_dispatch_keyset.
constexpr DispatchKeySet math_dispatch_keyset = backend_dispatch_keyset | autograd_dispatch_keyset;

DispatchKeySet getRuntimeDispatchKeySet(DispatchKey t) {
  TORCH_INTERNAL_ASSERT(t != DispatchKey::Undefined);
  switch (t) {
    case DispatchKey::Autograd:
      return autograd_dispatch_keyset;
    case DispatchKey::Math:
      return math_dispatch_keyset;
    default:
      return DispatchKeySet(t);
  }
}

DispatchKeySet getBackendKeySetFromAutograd(DispatchKey t) {
#define DEFINE_CASE(dk) \
  case DispatchKey::Autograd##dk: \
    return DispatchKeySet(DispatchKey::dk);

  switch (t) {
    FOR_EACH_BACKEND_DISPATCH_KEY_WITH_AUTOGRAD_KEY(DEFINE_CASE)
    case DispatchKey::AutogradOther:
      return autogradother_backends;
    default:
      return DispatchKeySet();
  }
#undef DEFINE_CASE
}

bool isIncludedInAlias(DispatchKey k, DispatchKey alias) {
  return k != DispatchKey::Undefined && getRuntimeDispatchKeySet(alias).has(k);
}

std::string toString(DispatchKeySet ts) {
  std::stringstream ss;
  ss << ts;
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, DispatchKeySet ts) {
  if (ts.empty()) {
    os << "DispatchKeySet()";
    return os;
  }
  os << "DispatchKeySet(";
  DispatchKey tid;
  bool first = true;
  while ((tid = ts.highestPriorityTypeId()) != DispatchKey::Undefined) {
    if (!first) {
      os << ", ";
    }
    os << tid;
    ts = ts.remove(tid);
    first = false;
  }
  os << ")";
  return os;
}

}
