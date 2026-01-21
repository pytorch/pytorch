#pragma once

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>

#include <cstdint>
#include <ostream>

// Memory format is not the property of a Tensor. It is the way to tell an
// operator how the result should be organized in memory and nothing more. That
// means memory format should never be used as return value for any tensor state
// interrogation functions (internally and externally).
//
// Possible options are:
//  Preserve:
//    If any of the input tensors is in channels_last format, operator output
//    should be in channels_last format
//
//  Contiguous:
//    Regardless of input tensors format, the output should be contiguous
//    Tensor.
//
//  ChannelsLast:
//    Regardless of input tensors format, the output should be in channels_last
//    format.

namespace c10 {

enum class MemoryFormat : int8_t {
  Contiguous,
  Preserve,
  ChannelsLast,
  ChannelsLast3d,
  NumOptions
};

inline MemoryFormat get_contiguous_memory_format() {
  return MemoryFormat::Contiguous;
}

} // namespace c10

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)
using c10::get_contiguous_memory_format;
using c10::MemoryFormat;
HIDDEN_NAMESPACE_END(torch, headeronly)
