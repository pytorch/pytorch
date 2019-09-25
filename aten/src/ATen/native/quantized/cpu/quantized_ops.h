#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

using qrelu_fn = void (*)(const at::Tensor& /*qx*/, at::Tensor& /*qy*/);
using qadd_fn =
    void (*)(Tensor& /*out*/, const Tensor& /*self*/, const Tensor& /*other*/);
using qmaxpool_2d_fn = void (*)(
    const Tensor& qx,
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
    Tensor& qy);
using qadaptive_avg_pool2d_fn = void (*)(
    const Tensor& qx,
    Tensor& qy,
    int64_t b,
    int64_t sizeD,
    int64_t isizeH,
    int64_t isizeW,
    int64_t osizeH,
    int64_t osizeW,
    int64_t istrideB,
    int64_t istrideD,
    int64_t istrideH,
    int64_t istrideW);

using qavg_pool2d_fn = void (*)(
    const Tensor& qx,
    Tensor& qy,
    int64_t b,
    int64_t nInputPlane,
    int64_t inputWidth,
    int64_t inputHeight,
    int64_t outputWidth,
    int64_t outputHeight,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);

using qupsample_bilinear2d_fn = void (*)(
    Tensor& output,
    const Tensor& input,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    bool align_corners);

using qcat_nhwc_fn = Tensor (*)(
    const c10::List<Tensor>& qxs,
    int64_t dim,
    double scale,
    int64_t zero_point);
using qtopk_fn = void(*)(Tensor&, Tensor&, const Tensor&, int64_t, int64_t, bool, bool);

// using qavg_pool2d_fn
DECLARE_DISPATCH(qrelu_fn, qrelu_stub);
DECLARE_DISPATCH(qrelu_fn, qrelu6_stub);
DECLARE_DISPATCH(qadd_fn, qadd_stub);
DECLARE_DISPATCH(qadd_fn, qadd_relu_stub);
DECLARE_DISPATCH(qmaxpool_2d_fn, qmaxpool_2d_nhwc_stub);
DECLARE_DISPATCH(qadaptive_avg_pool2d_fn, qadaptive_avg_pool2d_nhwc_stub);
DECLARE_DISPATCH(qavg_pool2d_fn, qavg_pool2d_nhwc_stub);
DECLARE_DISPATCH(qupsample_bilinear2d_fn, qupsample_bilinear2d_nhwc_stub);
DECLARE_DISPATCH(qcat_nhwc_fn, qcat_nhwc_stub);
DECLARE_DISPATCH(qcat_nhwc_fn, qcat_relu_nhwc_stub);
DECLARE_DISPATCH(qtopk_fn, qtopk_stub);

} // namespace native
} // namespace at
