#pragma once

#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <iostream>

namespace c10 {

/**
 * Casting option for converting between types.
 */
enum class Casting : int8_t { No, Promote, Unsafe, NumOptions };

static inline bool canCast(ScalarType from, ScalarType to, Casting casting) {
  switch (casting) {
    case Casting::No:
      return (to == from);
    case Casting::Promote:
      return (to == promoteTypes(from, to));
    case Casting::Unsafe:
      return true;
    default:
      throw std::runtime_error("Unknown casting");
  }
}

inline std::ostream& operator<<(std::ostream& stream, Casting casting) {
  switch (casting) {
    case Casting::No:
      return stream << "no";
    case Casting::Promote:
      return stream << "promote";
    case Casting::Unsafe:
      return stream << "unsafe";
    default:
      throw std::runtime_error("invalid casting value");
  }
}

static inline optional<Casting> parseCastingValue(const std::string& value) {
  static const std::string kScope = "Casting::";
  int val_pos =
      (value.compare(0, kScope.length(), kScope) == 0) ? kScope.length() : 0;
  if (value.compare(val_pos, string::npos, "No") == 0) {
    return Casting::No;
  } else if (value.compare(val_pos, string::npos, "Promote") == 0) {
    return Casting::Promote;
  } else if (value.compare(val_pos, string::npos, "Unsafe") == 0) {
    return Casting::Unsafe;
  }
  return nullopt;
}

static inline optional<Casting> parsePyCastingValue(const std::string& value) {
  if (value == "no") {
    return Casting::No;
  } else if (value == "promote") {
    return Casting::Promote;
  } else if (value == "unsafe") {
    return Casting::Unsafe;
  }
  return nullopt;
}

} // namespace c10
