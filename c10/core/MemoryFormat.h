#pragma once

#include <c10/core/Backend.h>
#include <c10/util/Exception.h>

#include <iostream>

// Memory format is not the property of a Tensor. It is the way to tell an
// operator how the result should be organized in memory and nothing more. That
// means memory format should never be used as return value for any tensor state
// interrogation functions (internally and externally).
//
// Possible options are:
//  Any:
//    An operator can return Tensor with any memory format. This describes the
//    current behavior of operators.
//
//  Preserve:
//    If any of the input tensors is in channels_last format, operator output
//    should be in channels_last format
//
//  Contiguous:
//    Regardless of input tensors format, the output should be contiguous Tensor.
//
//  ChannelsLast:
//    Regardless of input tensors format, the output should be in channels_last format.


namespace c10 {
enum class MemoryFormat : int8_t { Any, Preserve, Contiguous, ChannelsLast };

inline std::ostream& operator<<(
    std::ostream& stream,
    at::MemoryFormat memory_format) {
  switch (memory_format) {
    case MemoryFormat::Any:
      return stream << "Any";
    case MemoryFormat::Preserve:
      return stream << "Preserve";
    case MemoryFormat::Contiguous:
      return stream << "Contiguous";
    case MemoryFormat::ChannelsLast:
      return stream << "ChannelsLast";
    default:
      AT_ERROR("Unknown memory format");
  }
}

} // namespace c10
