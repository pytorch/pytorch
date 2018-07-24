#include "ATen/ATen.h"

namespace at { namespace native {

static inline void checkInputs(const Tensor& self, const Tensor& A, bool batched) {
  if (batched) {
    if (A.size(-1) != A.size(-2)) {
      AT_ERROR("A must be batches of square matrices, "
          "but they are %lld by %lld matrices",
          (long long)A.size(-1), (long long)A.size(-2));
    } else if (A.size(-1) != self.size(-2)) {
      AT_ERROR("incompatible matrix sizes for matmul: each a "
          "matrix is %llu by %lld but each b matrix is %lld by %lld.",
          (long long)A.size(-1), (long long)A.size(-1),
          (long long)self.size(-2), (long long)self.size(-1));
    }
  } else {
    if (A.dim() != 2) {
      AT_ERROR("A should have 2 dimensions, but has %d",
          A.dim());
    } else if (self.dim() != 1 && self.dim() != 2) {
      AT_ERROR("B should have 1 or 2 dimensions, but has %d",
          self.dim());
    } else if (A.size(0) != A.size(1)) {
      AT_ERROR("A must be a square matrix, but is %lld by %lld",
          (long long)A.size(0), (long long)A.size(1));
    } else if (A.size(0) != self.size(0)) {
      AT_ERROR("A,B size incompatible - A has %ld "
          "rows, B has %ld cols", A.size(0), self.size(0));
    }
  }
}

static inline void checkErrors(std::vector<int64_t> infos) {
  for (size_t i = 0; i < infos.size(); i++) {
    auto info = infos[i];
    if (info < 0) {
      AT_ERROR("gesv: For batch %lld: Argument %lld has illegal value",
          (long long)i, -info);
    } else if (info > 0) {
      AT_ERROR("gesv: For batch %lld: U(%lld,%lld) is zero, singular U.",
          (long long)i, info, info);
    }
  }
}

}}  // namespace at::native
