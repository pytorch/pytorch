#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

using _max_fn = void (*)(const Tensor&, const Tensor&);
DECLARE_DISPATCH(_max_fn, _max_stub);

} // namespace native
} // namespace at
