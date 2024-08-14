#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/ExpandUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/add_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/mul_native.h>
#endif

#if !AT_ONEDNN_ENABLED()

namespace at {
namespace native {

Tensor& mkldnn_add_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& result
    ) {
  TORCH_CHECK(false, "mkldnn_add_out: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_add(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  TORCH_CHECK(false, "mkldnn_add: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_add_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  TORCH_CHECK(false, "mkldnn_add_: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_mul_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_CHECK(false, "mkldnn_mul_out: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_mul(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "mkldnn_mul: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_mul_(Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "mkldnn_mul_: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_ONEDNN_ENABLED

#include <ATen/native/onednn/ONEDNNCommon.h>

namespace at {
namespace native {

static Tensor emptyBinaryOp(const Tensor& self, const Tensor& other) {
  if (!self.requires_grad() && !other.requires_grad()) {
    auto out_size = infer_size(self.sizes(), other.sizes());
    auto out_dtype = promoteTypes(
        c10::typeMetaToScalarType(self.dtype()),
        c10::typeMetaToScalarType(other.dtype()));
    TORCH_CHECK(
        self.device() == other.device(),
        "Expected same device for binary mkldnn op");
    return empty_mkldnn(
        out_size,
        out_dtype,
        self.options().layout_opt(),
        self.options().device_opt(),
        self.options().pinned_memory_opt());
  } else {
    TORCH_CHECK(
        false,
        "MKLDNN does not support Binary Ops with a 0-dimension Tensor in training");
  }
}

Tensor& mkldnn_add_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& result
    ) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor& y = itensor_from_mkldnn(other);

  ideep::tensor& z = itensor_from_mkldnn(result);
  if (result.is_same(other)) {
    const std::vector<float> scales{alpha.to<float>(), 1.0};
    ideep::sum::compute(scales, {y, x}, z);
  } else {
    const std::vector<float> scales{1.0, alpha.to<float>()};
    ideep::sum::compute(scales, {x, y}, z);
  }

  return result;
}

Tensor mkldnn_add(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  if (self.numel() == 0 || other.numel() == 0) {
    return emptyBinaryOp(self, other);
  }

  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor& y = itensor_from_mkldnn(other);

  ideep::tensor z;
  const std::vector<float> scales{1.0, alpha.to<float>()};
  ideep::sum::compute(scales, {x, y}, z);

  return new_with_itensor_mkldnn(std::move(z), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor& mkldnn_add_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return native::mkldnn_add_out(self, other, alpha, self);
}

Tensor& mkldnn_mul_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_CHECK(result.sizes() == self.sizes(),
             "mkldnn_mul_out: the output size should be same as input size");
  ideep::tensor& z = itensor_from_mkldnn(result);
  ideep::tensor& x = itensor_from_mkldnn(self);

  // for zero_dim tensor
  if (other.ndimension() == 0) {
    ideep::eltwise_forward::compute(
      x, z, ideep::algorithm::eltwise_linear,
      ideep::prop_kind::forward_inference, /*alpha*/ other.item().to<float>());

    return result;
  } else {
    TORCH_CHECK(self.sizes() == other.sizes(),
               "mkldnn_mul_out: currently mkldnn not support broadcasting");
    ideep::tensor y = itensor_from_mkldnn(other);
    ideep::binary::compute(x, y, z, dnnl::algorithm::binary_mul);

    return result;
  }
}

Tensor mkldnn_mul(const Tensor& self, const Tensor& other) {
  if (self.numel() == 0 || other.numel() == 0) {
    return emptyBinaryOp(self, other);
  }
  Tensor result = empty_mkldnn(self.sizes(), optTypeMetaToScalarType(self.options().dtype_opt()),
                               self.options().layout_opt(), self.options().device_opt(),
                               self.options().pinned_memory_opt());
  return native::mkldnn_mul_out(self, other, result);
}

Tensor& mkldnn_mul_(Tensor& self, const Tensor& other) {
  return native::mkldnn_mul_out(self, other, self);
}

} // namespace native
} // namespace at

#endif // AT_ONEDNN_ENABLED
