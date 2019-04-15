#ifndef C10_UTIL_TYPE_H_
#define C10_UTIL_TYPE_H_

#include <cstddef>
#include <string>
#include <typeinfo>

#include "c10/macros/Macros.h"

namespace c10 {

/// Utility to demangle a C++ symbol name.
C10_API std::string demangle(const char* name);

/// Returns the printable name of the type.
template <typename T>
inline const char* demangle_type() {
#ifdef __GXX_RTTI
  static const auto& name = *(new std::string(demangle(typeid(T).name())));
  return name.c_str();
#else // __GXX_RTTI
  return "(RTTI disabled, cannot show name)";
#endif // __GXX_RTTI
}

} // namespace c10

#endif // C10_UTIL_TYPE_H_
