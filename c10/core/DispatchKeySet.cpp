#include <c10/core/DispatchKeySet.h>

namespace c10 {

RuntimeDispatchKeySet getRuntimeDispatchKeySet(DispatchKey t) {
  switch (t) {
    case DispatchKey::Autograd:
      return autograd_dispatch_keyset;
    case DispatchKey::Math:
      return math_dispatch_keyset;
    case DispatchKey::DefaultBackend:
      return backend_dispatch_keyset;
    default:
      return RuntimeDispatchKeySet(t);
  }
}

// for a given autograd key, return the (guaranteed nonempty) set of associated backend keys.
// for a non-autograd key, return the empty keyset.
RuntimeDispatchKeySet getBackendKeySetFromAutograd(DispatchKey t) {
  switch (t) {
    case DispatchKey::AutogradCPU:
      return RuntimeDispatchKeySet(DispatchKey::CPU);
    case DispatchKey::AutogradCUDA:
      return RuntimeDispatchKeySet(DispatchKey::CUDA);
    case DispatchKey::AutogradXLA:
      return RuntimeDispatchKeySet(DispatchKey::XLA);
    case DispatchKey::AutogradPrivateUse1:
      return RuntimeDispatchKeySet(DispatchKey::PrivateUse1);
    case DispatchKey::AutogradPrivateUse2:
      return RuntimeDispatchKeySet(DispatchKey::PrivateUse2);
    case DispatchKey::AutogradPrivateUse3:
      return RuntimeDispatchKeySet(DispatchKey::PrivateUse3);
    case DispatchKey::AutogradOther:
      return autogradother_backends;
    default:
      return RuntimeDispatchKeySet();
  }
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
