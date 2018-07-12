#include <cmath>
#include <float.h>

#include <atomic>
#include "THTensor.hpp"
#include "THVector.h"
#include "generic/simd/simd.h"

#include "THBlas.h"
#include "THLapack.h"
#include "THRandom.h"
#include "THTensorDimApply.h"
#include "THMath.h"

#include "generic/THTensor.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THTensor.cpp"
#include "THGenerateHalfType.h"

#include "generic/THTensorCopy.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THTensorCopy.cpp"
#include "THGenerateHalfType.h"

#include "generic/THTensorRandom.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THTensorMath.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THTensorConv.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THTensorLapack.cpp"
#include "THGenerateFloatTypes.h"

#include <numeric>

void THTensor_free(THTensor *self)
{
  if(!self)
    return;

  if(self->flag & TH_TENSOR_REFCOUNTED)
  {
    if(--self->refcount == 0)
    {
      THFree(self->size);
      THFree(self->stride);
      if(self->storage)
        THStorage_free(self->storage);
      self->refcount.~atomic<int>();
      THFree(self);
    }
  }
}

// On a high level,
// 1. separate oldshape chunks of dimensions, where the dimensions are
//    ``contiguous'' in each chunk, i.e., oldstride[i] = oldshape[i+1] * oldstride[i+1]
// 2. newshape must be able to be separated into same number of chunks as oldshape was separated into,
//    where each chunk of newshape has matching ``numel'', i.e., number of subspaces,
//    as the corresponding chunk of oldshape.
at::optional<std::vector<int64_t>>
THTensor_compute_stride(at::IntList oldshape, at::IntList oldstride, at::IntList newshape) {
  if (oldshape.empty()) {
    return std::vector<int64_t>(newshape.size(), 1);
  }

  // NOTE: stride is arbitrary is somewhat arbitrary in the numel() == 0 case;
  // to match NumPy behavior we copy the strides if the size matches, otherwise
  // we use the stride as if it were computed via resize.
  // This could perhaps be combined with the below code, but the complexity didn't seem worth it.
  int64_t numel = std::accumulate(oldshape.begin(), oldshape.end(), 1, std::multiplies<int64_t>());
  if (numel == 0 && oldshape.equals(newshape)) {
    return std::vector<int64_t>(oldstride);
  }

  std::vector<int64_t> newstride(newshape.size());
  if (numel == 0) {
    int64_t view_numel = 1;
    for (int64_t view_d = newshape.size() - 1; view_d >= 0; view_d--) {
      if (view_d == newshape.size() - 1) {
        newstride[view_d] = 1;
      } else {
        newstride[view_d] = std::max<int64_t>(newshape[view_d+1], 1) * newstride[view_d+1];
      }
    }
    return newstride;
  }

  int64_t view_d = newshape.size() - 1;
  // stride for each subspace in the chunk
  int64_t chunk_base_stride = oldstride.back();
  // numel in current chunk
  int64_t tensor_numel = 1;
  int64_t view_numel = 1;
  for (int64_t tensor_d = oldshape.size() - 1; tensor_d >= 0; tensor_d--) {
    tensor_numel *= oldshape[tensor_d];
    // if end of tensor size chunk, check view
    if ((tensor_d == 0) ||
        (oldshape[tensor_d - 1] != 1 && oldstride[tensor_d - 1] != tensor_numel * chunk_base_stride)) {
      while (view_d >= 0 && (view_numel < tensor_numel || newshape[view_d] == 1)) {
        newstride[view_d] = view_numel * chunk_base_stride;
        view_numel *= newshape[view_d];
        view_d--;
      }
      if (view_numel != tensor_numel) {
        return at::nullopt;
      }
      if (tensor_d > 0) {
        chunk_base_stride = oldstride[tensor_d - 1];
        tensor_numel = 1;
        view_numel = 1;
      }
    }
  }
  if (view_d != -1) {
    return at::nullopt;
  }
  return newstride;
}
