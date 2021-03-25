#import <ATen/native/metal/MetalConvolution.h>
#import <ATen/native/metal/MetalPrepackOpContext.h>

#include <torch/script.h>

namespace at {
namespace native {
namespace metal {
namespace mpscnn {

Tensor conv2d(
    const Tensor& input, // metal
    const Tensor& weight, // cpu
    const c10::optional<at::Tensor>& bias, // cpu
    const Conv2DParams& params,
    NeuronType t = NeuronType::None);

// conv2d with prepacked weights
Tensor conv2d(const Tensor& input, Conv2dOpContext& context);

Tensor max_pool2d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode);

Tensor global_avg_pool2d(const Tensor& input, IntArrayRef output_size);

Tensor relu(const Tensor& input);

Tensor& relu_(Tensor& input);

Tensor sigmoid(const Tensor& input);

Tensor& hardsigmoid_(Tensor& input);

Tensor& hardtanh_(Tensor& input, Scalar min_val, Scalar max_val);

Tensor& hardswish_(Tensor& input);

Tensor t(const Tensor& input);

Tensor view(const Tensor& input, IntArrayRef size);

Tensor reshape(const Tensor& input, IntArrayRef shape);

Tensor addmm(const Tensor& bias, const Tensor& input, const Tensor& weight);

Tensor add(const Tensor& input1, const Tensor& input2);

Tensor& add_(Tensor& input1, const Tensor& input2);

Tensor sub(const Tensor& input1, const Tensor& input2);

Tensor mul(const Tensor& input1, const Tensor& input2);

Tensor log_softmax_int(const Tensor& input);

Tensor upsample_nearest2d_vec(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    c10::optional<ArrayRef<double>> scale_factors);

Tensor flatten_using_ints(const Tensor & input, int64_t start_dim, int64_t end_dim);

Tensor cat(const TensorList tensors, int64_t dim);

Tensor copy_to_host(const Tensor& input);

} // namespace mpscnn
} // namespace metal
} // namespace native
} // namespace at
