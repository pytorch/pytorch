#ifndef COMPUTATION_CLIENT_DEBUG_MACROS_H_
#define COMPUTATION_CLIENT_DEBUG_MACROS_H_

#include <iostream>

#include "lazy_tensors/statusor.h"

template <typename T>
T ConsumeValue(lazy_tensors::StatusOr<T>&& status) {
  CHECK(status.status().ok());
  return status.ConsumeValueOrDie();
}

#endif  // COMPUTATION_CLIENT_DEBUG_MACROS_H_
