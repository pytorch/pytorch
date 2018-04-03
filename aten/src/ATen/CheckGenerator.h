#pragma once

#include "ATen/Error.h"
#include "ATen/Generator.h"
#include "ATen/Utils.h"

namespace at {

template <typename T>
static inline T * check_generator(Generator * expr, Generator * defaultValue) {
  if (!expr)
    expr = defaultValue;
  if(auto result = dynamic_cast<T*>(expr))
    return result;
  AT_ERROR("Expected a '%s' but found '%s'", typeid(T).name(), typeid(expr).name());
}

} // namespace at
