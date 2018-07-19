#pragma once

#include <cstddef>
#include <string>
#include <typeinfo>

namespace at {
/// Utility to demangle a C++ symbol name.
std::string demangle(const char* name);

/// Returns the printable name of the type.
template <typename T>
inline const char* demangle_type() {
#ifdef __GXX_RTTI
  static const std::string name = demangle(typeid(T).name());
  return name.c_str();
#else // __GXX_RTTI
  return "(RTTI disabled, cannot show name)";
#endif // __GXX_RTTI
}

std::string get_backtrace(
    size_t frames_to_skip = 0,
    size_t maximum_number_of_frames = 64,
    bool skip_python_frames = true);
} // namespace at
