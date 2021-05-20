#include <c10/core/Scalar.h>
#include <ATen/native/DispatchStub.h>

// fwd:
namespace at {
class TensorIteratorBase;
class Tensor;
}

namespace at {
namespace native {

using histc_fn = void(*)(TensorIteratorBase& iter, Scalar min, Scalar max, const Tensor& result);

DECLARE_DISPATCH(histc_fn, histc_stub);

}}  // namespace at::native
