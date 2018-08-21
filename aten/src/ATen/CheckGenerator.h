#pragma once

#include "ATen/Generator.h"
#include "ATen/Utils.h"
#include "ATen/core/Error.h"

namespace at {

template <typename T>
static inline T * check_generator(Generator * expr, Generator * defaultValue) {
  if (!expr)
    expr = defaultValue;
  if(auto result = dynamic_cast<T*>(expr))
    return result;
  AT_ERROR("Expected a '", typeid(T).name(), "' but found '", typeid(expr).name(), "'");
}

} // namespace at
