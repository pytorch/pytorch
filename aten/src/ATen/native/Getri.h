#include "ATen/ATen.h"

namespace at {
namespace native {

static inline void checkInputs(const Tensor& self) {
  AT_CHECK(self.size(-1) == self.size(-2),
           "input matrix must be a batch of square matrices, "
           "but they are %lld by %lld matrices",
           (long long)self.size(-2), (long long)self.size(-1));
}

static inline void checkErrors(std::vector<int64_t> infos) {
  for (size_t i = 0; i < infos.size(); i++) {
    auto info = infos[i];
    if (info < 0) {
      AT_ERROR("getri: For batch %lld: Argument %lld has illegal value.",
               (long long)i, -info);
    } else if (info > 0) {
      AT_ERROR("getri: For batch %lld: U(%lld,%lld) is zero, singular U.",
               (long long)i, info, info);
    }
  }
}

} // namespace native
} // namespace at
