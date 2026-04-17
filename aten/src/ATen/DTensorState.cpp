#include <ATen/DTensorState.h>

namespace at {

namespace {
thread_local bool kDTensorAllowImplicitReplication = false;
}

bool get_dtensor_allow_implicit_replication() {
  return kDTensorAllowImplicitReplication;
}

void set_dtensor_allow_implicit_replication(bool enabled) {
  kDTensorAllowImplicitReplication = enabled;
}

} // namespace at
