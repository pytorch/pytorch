#pragma once

#include "ATen/Generator.h"
#include "ATen/Utils.h"

namespace at {

template <typename T>
static inline T * check_generator(Generator* expr) {
  if(auto result = dynamic_cast<T*>(expr))
    return result;
  runtime_error("Expected a '%s' but found '%s'", typeid(T).name(), typeid(expr).name());
}

} // namespace at
