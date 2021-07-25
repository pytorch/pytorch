#include <ATen/ATen.h>

#include <ATen/native/DispatchStub.h>
#include <c10/util/Exception.h>

namespace at {
namespace native {

static inline Tensor roll_common(const Tensor& self, IntArrayRef shifts, IntArrayRef dims) {
  TORCH_CHECK(shifts.size() > 0, "`shifts` required");
  if (dims.size() == 0 && shifts.size() == 1) {
    auto flattened = self.contiguous().view(self.numel());
    return roll(flattened, shifts[0], 0).view(self.sizes());
  }
  TORCH_CHECK(
    shifts.size() == dims.size(),
    "shifts and dimensions must align. shifts: ", shifts.size(), ", dims:", dims.size()
  );
  AT_ASSERT(dims.size() > 1);
  auto tail_shifts = shifts.slice(1);
  auto tail_dims = dims.slice(1);
  auto first_dim_rolled = roll(self, shifts[0], dims[0]);
  return at::roll(first_dim_rolled, tail_shifts, tail_dims);
}

using flip_fn = void(*)(TensorIterator &, const bool);
DECLARE_DISPATCH(flip_fn, flip_stub);

}}  // namespace at::native
