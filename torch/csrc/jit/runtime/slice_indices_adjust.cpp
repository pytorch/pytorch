#include <torch/csrc/jit/runtime/slice_indices_adjust.h>

#include <c10/util/Exception.h>
#include <cstdint>

namespace torch {
namespace jit {

int64_t slice_indices_adjust(
    int64_t length,
    int64_t* start,
    int64_t* stop,
    int64_t step) {
  TORCH_CHECK(step != 0, "List slice should have non-zero step")
  TORCH_CHECK(step >= -INT64_MAX, "List slice step is out of bounds")

  // Comes from PySlice_Unpack.
  if (*start == INT64_MAX) {
    *start = (step < 0) ? INT64_MAX : 0;
  }
  if (*stop == INT64_MAX) {
    *stop = (step < 0) ? INT64_MIN : INT64_MAX;
  }

  // Comes from PySlice_AdjustIndices.
  if (*start < 0) {
    *start += length;
    if (*start < 0) {
      *start = (step < 0) ? -1 : 0;
    }
  } else if (*start >= length) {
    *start = (step < 0) ? length - 1 : length;
  }

  if (*stop < 0) {
    *stop += length;
    if (*stop < 0) {
      *stop = (step < 0) ? -1 : 0;
    }
  } else if (*stop >= length) {
    *stop = (step < 0) ? length - 1 : length;
  }

  if (step < 0) {
    if (*stop < *start) {
      return (*start - *stop - 1) / (-step) + 1;
    }
  } else {
    if (*start < *stop) {
      return (*stop - *start - 1) / step + 1;
    }
  }
  return 0;
}

} // namespace jit
} // namespace torch
