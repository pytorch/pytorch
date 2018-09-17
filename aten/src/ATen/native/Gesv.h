#include "ATen/ATen.h"

namespace at { namespace native {

static inline void checkInputs(const Tensor& self, const Tensor& A) {
  if (A.size(-1) != A.size(-2)) {
    AT_ERROR("A must be batches of square matrices, "
        "but they are ", A.size(-1), " by ", A.size(-2), " matrices",
        (long long)A.size(-1), (long long)A.size(-2));
  }
  if (A.size(-1) != self.size(-2)) {
    AT_ERROR("Incompatible matrix sizes for matmul: each A "
        "matrix is ", A.size(-1), " by ", A.size(-1),
        " but each b matrix is ", self.size(-2), " by ", self.size(-1));
  }
}

static inline void checkErrors(std::vector<int64_t>& infos) {
  for (size_t i = 0; i < infos.size(); i++) {
    auto info = infos[i];
    if (info < 0) {
      AT_ERROR("gesv: For batch ", i, ": Argument ", -info, " has illegal value.");
    } else if (info > 0) {
      AT_ERROR("gesv: For batch ", i, ": U(", info, ",", info, ") is zero, singular U.");
    }
  }
}

}}  // namespace at::native
