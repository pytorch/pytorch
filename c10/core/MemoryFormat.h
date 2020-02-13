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
enum class MemoryFormat : int8_t { Contiguous, Preserve, ChannelsLast, ChannelsLast3d };

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
    case MemoryFormat::ChannelsLast3d:
      return stream << "ChannelsLast3d";
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

// Note [Ambiguous is_channels_last_strides]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The flaw of carrying memory_format implicitly through strides is very hard
// to WAR properly. issue #24090
// Without the history of permutation, we can't infer the memory_format of a
// tensor from the snapshot of its size & stride
// e.g.
//
// 1. We can NOT specify the memory_format of N111 tensor through strides in a
//  meaningful way;
//
// 2. Two path that ended up with identical size/stride
//  N11W contiguous tensor sliced at w-dimension becomes [N,1,1,1]@[W,W,W,W]
//  NC11 channels_last tensor sliced at c-dimension becomes [N,1,1,1]@[C,C,C,C]
//    So if we see a tensor [N,1,1,1]@[X,X,X,X], there's no way for us to infer
//    the memory_format of the original tensor.
//
// Due to the limitations, our temporary WAR `is_channels_last_strides` does the
// best effort to infer whether the original memory_format of a tensor is
// at::MemoryFormat::ChannelsLast. The two objectives of this function (ordered
// by their importance):
//   1. Ensure that normal shape manipulation does not accidentally change the
//      MemoryFormat of an existing tensor.
//   2. Allows user to mark MemoryFormat::ChannelsLast to tensors;
//
// The function does so via checking strides of the tensor, including strides of
// size-1 dimensions. Although conventionally PyTorch implies no restriction on
// trivial stride (stride for size-1 dimension).
//
// Note that this approach is a compromise. We did not solve the problem
// completely. Many cases we will not be able to infer the correct memory
// format.
// The implementation of `is_channels_last_strides` is to serve the objectives:
// MemoryFormat::ChannelsLast has to be explicitly opted-in (no accidental
// conversion); Best effort to maintain the ChannelsLast flag.
//
// Due to the fact that this is not a bulletproof solution, through testing
// (aten/src/ATen/test/memory_format_test.cpp)
//   a. we ensure that the common tasks are supported;
//   a. we identify corner cases where the implementation compromises on.
//
// By the time accumulated permutation is enabled to replace implicit
// memory_foramt through strides, we should be updating our tests and fix the
// issues in our tests.
inline bool is_channels_last_strides(const IntArrayRef sizes, const IntArrayRef strides) {
  if (sizes.size() == 4) {
    int64_t min = 0;
    // special case for trivial C dimension. default to NCHW
    if (strides[1]==0) {
      return false;
    }
    for (auto& d : {1, 3, 2, 0}) {
      if (sizes[d] == 0) {
        return false;
      }
      if (strides[d] < min) {
        return false;
      }
      // Fallback to NCHW as default layout for ambiguous cases
      // This is the flaw of implicit memory_format from strides.
      // N111 tensor with identical strides for size 1 dimension;
      // Two cases could lead us here:
      // a. N111 contiguous Tensor ([N,1,1,1]@[1,1,1,1])
      // b. N11W contiguous Tensor sliced on the W-dimension. ([N,1,1,1]@[W,W,W,W])
      if (d==0 && min==strides[1]) {
        return false;
      }
      // This is necessary to:
      // 1. distinguish the memory_format of N1H1;
      //     [H, 1, 1, 1] channels_last stride
      //     [H, H, 1, 1] contiguous stride
      // 2. permutation of 1C1W:
      //     [1, C, 1, H]@[HC, H, H, 1] transpose(1, 3)
      //     [1, H, 1, C]@[HC, 1, H, H] shouldn't be identified as channels_last
      min = strides[d];
      if (sizes[d] > 1) {
        min *= (sizes[d]-1);
      }
    }
    return true;
  }
  return false;
}

} // namespace c10
