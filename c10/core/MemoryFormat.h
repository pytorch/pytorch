#pragma once

#include <c10/core/Backend.h>
#include <c10/util/Exception.h>
#include <c10/util/ArrayRef.h>

#include <iostream>

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
//    Regardless of input tensors format, the output should be contiguous Tensor.
//
//  ChannelsLast:
//    Regardless of input tensors format, the output should be in channels_last format.


namespace c10 {
enum class MemoryFormat : int8_t { Contiguous, Preserve, ChannelsLast };

// If you are seeing this, it means that this call site was not checked if
// the memory format could be preserved, and it was switched to old default
// behaviour of contiguous
#define LEGACY_CONTIGUOUS_MEMORY_FORMAT c10::get_contiguous_memory_format()

C10_DEPRECATED inline MemoryFormat get_contiguous_memory_format() {
  return MemoryFormat::Contiguous;
}

inline std::ostream& operator<<(
    std::ostream& stream,
    at::MemoryFormat memory_format) {
  switch (memory_format) {
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

inline std::vector<int64_t> get_channels_last_strides(IntArrayRef sizes) {
  AT_ASSERT(sizes.size() == 4);
  std::vector<int64_t> strides(sizes.size());
  strides[1] = 1;
  strides[3] = sizes[1];
  strides[2] = strides[3] * sizes[3];
  strides[0] = strides[2] * sizes[2];
  return strides;
}

inline bool is_channels_last_strides(const IntArrayRef sizes, const IntArrayRef strides) {
  if (sizes.size() == 4) {
    int64_t min = 0;
    // special case for trivial C dimension. default to NCHW
    if (strides[1]==0) {
      return false;
    }
    for (auto& d : {1, 3, 2, 0}) {
      if (strides[d] < min) {
        return false;
      }
      // special case for N111, so we have NCHW as default layout;
      // There's no way to disambiguate the memory format: issue #24090
      if (d==0 && min==1) {
        return false;
      }
      // This is necessary to distinguish the memory_format of N1H1;
      // [H, 1, 1, 1] channels_last stride
      // [H, H, 1, 1] contiguous stride
      min = std::max(strides[d], sizes[d]);
    }
    return true;
  }
  return false;
}

} // namespace c10
