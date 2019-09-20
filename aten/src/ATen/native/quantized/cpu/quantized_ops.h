#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

using qrelu_fn = void (*)(const at::Tensor& /*qx*/, at::Tensor& /*qy*/);
using qadd_fn =
    void (*)(Tensor& /*out*/, const Tensor& /*self*/, const Tensor& /*other*/);
using qmaxpool_2d_fn =
    void (*)(const Tensor &qx,
             int64_t iC, // input/output channels
             int64_t iH,
             int64_t iW, // input sizes
             int64_t oH,
             int64_t oW, // output sizes
             int64_t kH,
             int64_t kW, // kernel size
             int64_t sH,
             int64_t sW, // strides
             int64_t pH,
             int64_t pW, // padding
             int64_t dH,
             int64_t dW, // dilation
             Tensor &qy
            );

DECLARE_DISPATCH(qrelu_fn, qrelu_stub);
DECLARE_DISPATCH(qrelu_fn, qrelu6_stub);
DECLARE_DISPATCH(qadd_fn, qadd_stub);
DECLARE_DISPATCH(qadd_fn, qadd_relu_stub);
DECLARE_DISPATCH(qmaxpool_2d_fn, qmaxpool_2d_nhwc_stub);

} // namespace native
} // namespace at