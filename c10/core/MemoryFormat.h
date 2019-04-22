#pragma once

#include <c10/core/Backend.h>
#include <c10/util/Exception.h>

#include <iostream>

namespace c10 {
enum class MemoryFormat : int8_t { Any, Preserve, Contiguous, ChannelsFirst };

inline std::ostream& operator<<(std::ostream& stream, at::MemoryFormat memory_format) {
  switch (memory_format) {
    case MemoryFormat::Any:
      return stream << "Any";
    case MemoryFormat::Preserve:
      return stream << "Preserve";
    case MemoryFormat::Contiguous:
      return stream << "Contiguous";
    case MemoryFormat::ChannelsFirst:
      return stream << "ChannelsFirst";
    default:
      AT_ERROR("Unknown memory format");
  }
}

} // namespace c10
