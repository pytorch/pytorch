#pragma once

namespace tlib {

#define TensorLib_assert(cond, ...) if (!cond) { tlib::runtime_error(__VA_ARGS__); }
void runtime_error(const char *format, ...);

} // tlib
