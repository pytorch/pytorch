#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorEvenMoreMath.cpp"
#else

#include <TH/generic/THTensorApply.hpp>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/MemoryOverlap.h>

// Finds non-zero elements of a tensor and returns their subscripts
void THTensor_(nonzero)(THLongTensor *subscript, THTensor *tensor)
{
  ptrdiff_t numel = 0;
  int64_t *subscript_data;
  int64_t i = 0;
#ifdef TH_REAL_IS_HALF
#define IS_NONZERO(val) (c10::Half(0)!=val)
#elif defined(TH_REAL_IS_BFLOAT16)
#define IS_NONZERO(val) (c10::BFloat16(0)!=val)
#else
#define IS_NONZERO(val) ((val)!=scalar_t(0))
#endif

  /* First Pass to determine size of subscripts */
  TH_TENSOR_APPLY(scalar_t, tensor,
                  if IS_NONZERO(*tensor_data) {
                    ++numel;
                  });
#ifdef DEBUG
  THAssert(numel <= LONG_MAX);
#endif
  THLongTensor_resize2d(subscript, numel, tensor->dim());
  if (numel <= 0) {
    return;
  }
  int64_t dimensions = tensor->dim();
  // +1 faster than additional condition check inside loop
  int64_t *sizes = new int64_t[dimensions+1];
  int64_t *idx = new int64_t[dimensions+1];
  int64_t *ii;
  int64_t *ss;
  std::fill(idx, idx+dimensions+1, 0);
  for (i = 0; i < dimensions; ++i) {
    sizes[dimensions - i - 1] = THTensor_(size)(tensor, i); // reverse order important
  }
  sizes[dimensions] = 0;
  /* Second pass populates subscripts */
  subscript_data = THLongTensor_data(subscript);
  auto subscript_strides = THTensor_stridesLegacyNoScalars(subscript);
  subscript_strides[0] -= subscript_strides[1] * tensor->dim();
  TH_TENSOR_APPLY(scalar_t, tensor,
                  if IS_NONZERO(*tensor_data) {
                    ii = idx + dimensions;
                    for (int64_t dim = dimensions - 1; dim >= 0; dim--) {
                      --ii;
                      *subscript_data = *ii;
                      subscript_data += subscript_strides[1];
                    }
                    subscript_data += subscript_strides[0];
                  }
                  ii = idx;
                  ss = sizes;
                  ++(*ii);
                  while (*ii == *ss) {
                    *ii = 0;
                    ++ii;
                    ++ss;
                    ++(*ii);
                  }
                );
  delete [] sizes;
  delete [] idx;

#undef IS_NONZERO
}
#endif /* TH_GENERIC_FILE */
