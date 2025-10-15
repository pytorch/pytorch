#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_softmax_native.h>         // for mkldnn_softmax
#endif

#if !AT_MKLDNN_ENABLED()


namespace at::native {

Tensor mkldnn_softmax(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  TORCH_CHECK(false, "mkldnn_softmax: ATen not compiled with MKLDNN support");
}

} // namespace at::native


#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native {

Tensor mkldnn_softmax(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  TORCH_CHECK(
      !half_to_float,
      "softmax with half to float conversion is not supported on Mkldnn");
  const int64_t wrapped_dim = maybe_wrap_dim(dim, self.dim());
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor y;
  ideep::softmax_forward::compute(x, y, wrapped_dim);
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

} // namespace at

#endif // AT_MKLDNN_ENABLED
