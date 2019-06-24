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
// In case the aten Tensor is a mkldnn tensor,
//   1. reshape first if input dim is greater than 2 and the reshape will cost a memory copy.
//   2. directly return the ideep tensor if the input dim is 2.
// In case the aten Tensor is a dense tensor, make sure:
//   1. the aten tensor is 2d, so that 3d*2d Linear is viewed as 2d*2d.
//   2. the aten tensor is contiguous.
inline ideep::tensor get_mkldnn_tensor(const at::Tensor& tensor) {
  if (tensor.is_mkldnn()) {
    if (tensor.dim() > 2) {
      // use reshape for mkldnn tensor
      auto x = tensor.reshape({-1, tensor.size(tensor.dim() - 1)});
      return itensor_from_mkldnn(x);
    }
    return itensor_from_mkldnn(tensor);
  } else {
    auto x = tensor.contiguous().view({-1, tensor.size(tensor.dim() - 1)});
    return itensor_view_from_dense(x);
  }
}

Tensor mkldnn_linear(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias) {
  TORCH_CHECK(self.dim() >= 2,
      "mkldnn_linear: input needs to has dim at least 2, input dim ", self.dim());

  const ideep::tensor& x = get_mkldnn_tensor(self);
  const ideep::tensor& w = itensor_from_mkldnn(weight);

  ideep::tensor y;
  if (bias.defined()) {
    const ideep::tensor b = itensor_from_mkldnn(bias);
    ideep::inner_product_forward::compute(x, w, b, y);
  } else {
    ideep::inner_product_forward::compute(x, w, y);
  }

  auto input_size = self.sizes();
  auto weight_size = weight.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight.size(0));

  if (self.is_mkldnn()) {
    if (self.dim() > 2) {
      return new_with_itensor_mkldnn(std::move(y), self.options()).reshape(output_size);
    }
    return new_with_itensor_mkldnn(std::move(y), self.options());
  } else {
    return mkldnn_to_dense(
        new_with_itensor_mkldnn(std::move(y), self.options())).view(output_size);
  }
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
