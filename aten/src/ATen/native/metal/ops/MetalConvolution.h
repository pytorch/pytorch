#import <ATen/native/metal/MetalConvParams.h>
#import <ATen/native/metal/MetalPrepackOpContext.h>
#include <c10/util/ArrayRef.h>

namespace at {
namespace native {
namespace metal {

Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups);

namespace prepack {
Tensor conv2d(const Tensor& input, Conv2dOpContext& context);
}

} // namespace metal
} // namespace native
} // namespace at
