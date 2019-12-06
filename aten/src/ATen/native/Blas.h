#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using addmv_stub_t = void (*)(Tensor&, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar);

DECLARE_DISPATCH(addmv_stub_t, addmv_stub);

}} // namespace at::native
