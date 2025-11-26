#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/TensorAccessor.h>

template <typename T, size_t N>
using Accessor_cpu = torch::headeronly::HeaderOnlyTensorAccessor<T, N>;

#if defined(__CUDACC__) || defined(__HIPCC__)
#define MAYBE_GLOBAL __global__

template <typename T, size_t N>
using Accessor_cuda = torch::headeronly::HeaderOnlyGenericPackedTensorAccessor<T, N, torch::headeronly::RestrictPtrTraits>;

#else
#define MAYBE_GLOBAL
#endif

template <template <typename, size_t> class Accessor, typename scalar_t>
MAYBE_GLOBAL void mv_tensor_accessor_kernel(Accessor<scalar_t, 1> resa, Accessor<scalar_t, 2> ma, Accessor<scalar_t, 1> va) {
  for (int64_t i = 0; i < resa.size(0); i++) {
    scalar_t val = 0;
    for (int64_t j = 0; j < ma.size(1); j++) {
      val += ma[i][j] * va[j];
    }
    resa[i] = val;
  }
}
