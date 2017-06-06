#pragma once

namespace tlib {

#define TensorLib_assert(cond, ...) if (!cond) { tlib::runtime_error(__VA_ARGS__); }

[[noreturn]]
void runtime_error(const char *format, ...);

template <typename T, typename Base>
static inline T* checked_cast(Base* expr) {
  if(auto result = dynamic_cast<T*>(expr))
    return result;
  runtime_error("Expected a '%s' but found '%s'",T::typeString(),expr->type().toString());
}

} // tlib
