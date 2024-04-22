#include <ATen/native/DispatchStub.h>
#include <c10/core/Scalar.h>

namespace at {
struct TensorIterator;

namespace native {

DECLARE_DISPATCH(void(*)(TensorIterator&, const Scalar&, const Scalar&, const Scalar&), arange_stub);
DECLARE_DISPATCH(void(*)(TensorIterator&, const Scalar&, const Scalar&, int64_t), linspace_stub);

}}  // namespace at::native
