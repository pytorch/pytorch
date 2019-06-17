#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor mkldnn_linear(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias) {
  AT_ERROR("mkldnn_linear: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

// Helper function for getting an ideep tensor out of an aten Tensor.
// Note in case the aten Tensor is a dense tensor, make sure:
//   1. the aten tensor is 2d, so that 3d*2d Linear is viewed as 2d*2d.
//   2. the aten tensor is contiguous.
inline ideep::tensor get_mkldnn_tensor(const at::Tensor& tensor) {
  if (tensor.is_mkldnn()) {
    return at::native::itensor_from_mkldnn(tensor);
  } else {
    auto x = tensor.contiguous().view({-1, tensor.size(tensor.dim() - 1)});
    return at::native::itensor_view_from_dense(x);
  }
}

Tensor mkldnn_linear(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias) {
  const ideep::tensor x = get_mkldnn_tensor(self);
  const ideep::tensor w = get_mkldnn_tensor(weight);

  ideep::tensor y;
  if (bias.defined()) {
    const ideep::tensor b = get_mkldnn_tensor(bias);
    ideep::inner_product_forward::compute(x, w, b, y);
  } else {
    ideep::inner_product_forward::compute(x, w, y);
  }

  if (self.is_mkldnn()) {
    return new_with_itensor_mkldnn(std::move(y), self.options());
  } else {
    auto input_size = self.sizes();
    auto weight_size = weight.sizes();
    std::vector<int64_t> output_size;
    output_size.insert(output_size.end(), input_size.begin(), input_size.end() - 1);
    output_size.push_back(weight.size(0));
    return mkldnn_to_dense(
        new_with_itensor_mkldnn(std::move(y), self.options())).view(output_size);
  }
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
