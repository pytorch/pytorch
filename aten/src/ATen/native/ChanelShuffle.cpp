#include <ATen/native/TensorTransformations.h>

#include <ATen/NativeFunctions.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <vector>

namespace at {
namespace native {

Tensor channel_shuffle(const Tensor& self, int64_t groups) {
  AT_ASSERTM(self.dim() == 4,
             "channel_shuffle expects 4D input, but got input with sizes ",self.sizes());
  int64_t b = self.size(0);
  int64_t c = self.size(1);
  int64_t h = self.size(2);
  int64_t w = self.size(3);
  AT_ASSERTM((c % groups) == 0,
             "Number of channels must be divisible gy groups. Got ",
             c, " channels and ", groups, " groups.");
  int64_t oc = c / groups;

  auto input_reshaped = self.reshape({b, groups, oc, h, w});
  return input_reshaped.permute({0 /* b */, 2 /* oc */, 1 /* groups */, 3 /* h */, 4 /* w */})
                       .contiguous()
                       .reshape({b, c, h, w});
}

}} // namespace at::native
