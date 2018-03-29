#include "ATen/ATen.h"

namespace at { namespace native {

static inline void checkInputs(const Tensor& self, const Tensor& A) {
  if (A.size(-1) != A.size(-2)) {
    runtime_error("A must be batches of square matrices, "
        "but they are %llu by %llu matrices",
        (long long)A.size(-1), (long long)A.size(-2));
  }
  if (A.size(-1) != self.size(-2)) {
    runtime_error("Incompatible matrix sizes for matmul: each A "
        "matrix is %llu by %llu but each b matrix is %llu by %llu.",
        (long long)A.size(-1), (long long)A.size(-1),
        (long long)self.size(-2), (long long)self.size(-1));
  }
  if (A.ndimension() != self.ndimension()) {
    runtime_error("arguments have differing number "
        "of dimensions: got %llu and %llu",
        (long long)A.ndimension(), (long long)self.ndimension());
  }
  for (int64_t i = 0; i < A.ndimension() - 1; i++) {
    if (A.size(i) == self.size(i)) {
      continue;
    }
    runtime_error("Incompatible batch size at dimension %llu: "
        "got size %llu for A and %llu for b",
        i, (long long)A.size(i), self.size(i));
  }
}

static inline void checkErrors(std::vector<int64_t> infos) {
  for (size_t i = 0; i < infos.size(); i++) {
    auto info = infos[i];
    if (info < 0) {
      runtime_error("gesv: For batch %llu: Argument %llu has illegal value",
          (long long)i, -info);
    } else if (info > 0) {
      runtime_error("gesv: For batch %llu: U(%llu,%llu) is zero, singular U.",
          (long long)i, info, info);
    }
  }
}

}}  // namespace at::native
