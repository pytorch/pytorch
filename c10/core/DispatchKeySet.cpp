#include <c10/core/DispatchKeySet.h>

namespace c10 {

constexpr DispatchKeySet autograd_dispatch_keyset = DispatchKeySet({
  DispatchKey::AutogradCPU,
  DispatchKey::AutogradCUDA,
  DispatchKey::AutogradXLA,
  DispatchKey::AutogradPrivateUse1,
  DispatchKey::AutogradPrivateUse2,
  DispatchKey::AutogradPrivateUse3,
  DispatchKey::AutogradOther,
});

DispatchKeySet getRuntimeDispatchKeySet(DispatchKey t) {
  TORCH_INTERNAL_ASSERT(t != DispatchKey::Undefined);
  switch (t) {
    case DispatchKey::Autograd:
      return autograd_dispatch_keyset;
    case DispatchKey::Undefined:
     return DispatchKeySet();
   default:
     return DispatchKeySet(t);
  }
}

bool isIncludedInAlias(DispatchKey k, DispatchKey alias) {
  return k != DispatchKey::Undefined && getRuntimeDispatchKeySet(alias).has(k);
}

DispatchKeySet getRuntimeAutogradKeySet() {
  return autograd_dispatch_keyset;
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
