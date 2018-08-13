#include "ATen/ATen.h"

namespace at { namespace native {

static inline void checkInputs(const Tensor& self, const Tensor& A, bool batched) {
  if (batched) {
    if (A.size(-1) != A.size(-2)) {
      AT_ERROR("A must be batches of square matrices, "
          "but they are ", A.size(-1), " by ", A.size(-2), " matrices");
    } else if (A.size(-1) != self.size(-2)) {
      AT_ERROR("incompatible matrix sizes for matmul: each a "
          "matrix is ", A.size(-1), " by ", A.size(-1),
          " but each b matrix is ", self.size(-2), " by ", self.size(-1));
    }
  } else {
    if (A.dim() != 2) {
      AT_ERROR("A should have 2 dimensions, but has ", A.dim());
    } else if (self.dim() != 1 && self.dim() != 2) {
      AT_ERROR("B should have 1 or 2 dimensions, but has ", self.dim());
    } else if (A.size(0) != A.size(1)) {
      AT_ERROR("A must be a square matrix, but is ",
          A.size(0), " by ", A.size(1));
    } else if (A.size(0) != self.size(0)) {
      AT_ERROR("A,B size incompatible - A has ", A.size(0),
          " rows, B has ", self.size(0), " cols");
    }
  }
}

static inline void checkErrors(std::vector<int64_t> infos) {
  for (size_t i = 0; i < infos.size(); i++) {
    auto info = infos[i];
    if (info < 0) {
      AT_ERROR("gesv: For batch ", i, ": Argument ",
          -info, " has illegal value");
    } else if (info > 0) {
      AT_ERROR("gesv: For batch ", i, ": U(", info, ",", info,
          ") is zero, singular U.");
    }
  }
}

}}  // namespace at::native
