#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

using qrelu_fn = void (*)(const at::Tensor& /*qx*/, at::Tensor& /*qy*/);
using qadd_fn =
    void (*)(Tensor& /*out*/, const Tensor& /*self*/, const Tensor& /*other*/);

DECLARE_DISPATCH(qrelu_fn, qrelu_stub);
DECLARE_DISPATCH(qrelu_fn, qrelu6_stub);
DECLARE_DISPATCH(qadd_fn, qadd_stub);
DECLARE_DISPATCH(qadd_fn, qadd_relu_stub);

} // namespace native
} // namespace at