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

Tensor mkldnn_linear(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias) {
  TORCH_CHECK(self.dim() >= 2,
      "mkldnn_linear: input needs to has dim at least 2, input dim ", self.dim());
  TORCH_CHECK(self.is_mkldnn(),
      "mkldnn_linear: input needs to be mkldnn layout");
  TORCH_CHECK(weight.is_mkldnn() && bias.is_mkldnn(),
      "mkldnn_linear: weight and bias need to be mkldnn layout");

  // reshape first if input dim is greater than 2 and the reshape will cost a memory copy.
  auto self_reshaped = self.dim() > 2 ? self.reshape({-1, self.size(self.dim() - 1)}) : self;
  const ideep::tensor x = itensor_from_mkldnn(self_reshaped);
  const ideep::tensor w = itensor_from_mkldnn(weight);

  ideep::tensor y;
  if (bias.defined()) {
    const ideep::tensor b = itensor_from_mkldnn(bias);
    ideep::inner_product_forward::compute(x, w, b, y);
  } else {
    ideep::inner_product_forward::compute(x, w, y);
  }

  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight.size(0));

  if (self.dim() > 2) {
    return new_with_itensor_mkldnn(std::move(y), self.options()).reshape(output_size);
  }
  return new_with_itensor_mkldnn(std::move(y), self.options());
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
