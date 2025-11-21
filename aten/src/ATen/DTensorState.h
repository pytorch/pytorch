#pragma once

#include <c10/macros/Macros.h>

namespace at {

TORCH_API bool get_dtensor_allow_implicit_replication();
TORCH_API void set_dtensor_allow_implicit_replication(bool enabled);

struct DTensorAllowImplicitReplication {
  DTensorAllowImplicitReplication()
      : prev_dtensor_allow_implicit_replication_(
            get_dtensor_allow_implicit_replication()) {
    set_dtensor_allow_implicit_replication(true);
  }

  DTensorAllowImplicitReplication(const DTensorAllowImplicitReplication&) =
      delete;
  DTensorAllowImplicitReplication& operator=(
      const DTensorAllowImplicitReplication&) = delete;
  DTensorAllowImplicitReplication(DTensorAllowImplicitReplication&&) = delete;
  DTensorAllowImplicitReplication& operator=(
      DTensorAllowImplicitReplication&&) = delete;

  ~DTensorAllowImplicitReplication() {
    set_dtensor_allow_implicit_replication(
        prev_dtensor_allow_implicit_replication_);
  }

 private:
  bool prev_dtensor_allow_implicit_replication_;
};

} // namespace at
