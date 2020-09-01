#include <c10/core/DispatchKeySet.h>

namespace c10 {

static DispatchKeySet autograd_dispatch_keys{
  DispatchKey::Autograd,
  DispatchKey::AutogradXLA,
  DispatchKey::PrivateUse1_PreAutograd,
  DispatchKey::PrivateUse2_PreAutograd,
  DispatchKey::PrivateUse3_PreAutograd,
};

DispatchKeySet AutogradDispatchKeys() {
  return autograd_dispatch_keys;
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
