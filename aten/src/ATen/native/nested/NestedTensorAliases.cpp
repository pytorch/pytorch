#include <ATen/ATen.h>


namespace at::native {

// alias for to_padded_tensor in nested namespace
Tensor nested_to_padded_tensor(
    const Tensor& t,
    double padding,
    OptionalIntArrayRef output_size) {
    return t.to_padded_tensor(padding, output_size);
}

} // namespace at::native
