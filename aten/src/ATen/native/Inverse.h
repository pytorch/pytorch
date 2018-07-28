#include "ATen/ATen.h"

namespace at {
namespace native {

static inline void checkInputs(const Tensor& self) {
  AT_CHECK(self.size(-1) == self.size(-2),
           "input matrix must be a batch of square matrices, "
           "but they are ", self.size(-2), " by ", self.size(-1), " matrices");
}

static inline void checkErrors(std::vector<int64_t> infos, const char* name) {
  for (size_t i = 0; i < infos.size(); i++) {
    auto info = infos[i];
    if (info < 0) {
      AT_ERROR(name, ": For batch ", i, ": Argument ", -info, " has illegal value");
    } else if (info > 0) {
      AT_ERROR(name, ": For batch ", i, ": U(", info, ",", info, ") is zero, singular U.");
    }
  }
}

} // namespace native
} // namespace at
