#pragma once

#include <ATen/ScalarType.h>
#include <vector>

// NOTE: these functions are for compatibility into TH functions that takes sizes and strides.
// We should just write the TH functions that don't require this, but that involves:
// differentiating between nullptr/non-nullptr strides (the former "infers" strides).

namespace at {

static inline IntList get_intlist_stride_th(IntList strides) {
  if (strides.size() == 0) {
    // differentiating between nullptr/non-nullptr strides (the former "infers" strides)
    return IntList();
  } else {
    return strides;
  }
}

}
