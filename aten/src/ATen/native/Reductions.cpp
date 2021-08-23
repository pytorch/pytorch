#include <ATen/native/Reductions.h>

namespace at {
namespace native {

Tensor _max(const Tensor& self) {
  const auto out = at::empty({}, self.options());
  _max_stub(self.device().type(), self, out);
  return out;
}

DEFINE_DISPATCH(_max_stub);

} // namespace native
} // namespace at
