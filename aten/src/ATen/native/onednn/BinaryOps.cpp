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

Tensor& onednn_add_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& result
    ) {
  TORCH_CHECK(false, "onednn_add_out: ATen not compiled with ONEDNN support");
}

Tensor onednn_add(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  TORCH_CHECK(false, "onednn_add: ATen not compiled with ONEDNN support");
}

Tensor& onednn_add_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  TORCH_CHECK(false, "onednn_add_: ATen not compiled with ONEDNN support");
}

Tensor& onednn_mul_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_CHECK(false, "onednn_mul_out: ATen not compiled with ONEDNN support");
}

Tensor onednn_mul(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "onednn_mul: ATen not compiled with ONEDNN support");
}

Tensor& onednn_mul_(Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "onednn_mul_: ATen not compiled with ONEDNN support");
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
        "Expected same device for binary onednn op");
    return empty_onednn(
        out_size,
        out_dtype,
        self.options().layout_opt(),
        self.options().device_opt(),
        self.options().pinned_memory_opt());
  } else {
    TORCH_CHECK(
        false,
        "ONEDNN does not support Binary Ops with a 0-dimension Tensor in training");
  }
}

Tensor& onednn_add_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& result
    ) {
  ideep::tensor& x = itensor_from_onednn(self);
  ideep::tensor& y = itensor_from_onednn(other);

  ideep::tensor& z = itensor_from_onednn(result);
  if (result.is_same(other)) {
    const std::vector<float> scales{alpha.to<float>(), 1.0};
    ideep::sum::compute(scales, {y, x}, z);
  } else {
    const std::vector<float> scales{1.0, alpha.to<float>()};
    ideep::sum::compute(scales, {x, y}, z);
  }

  return result;
}

Tensor onednn_add(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  if (self.numel() == 0 || other.numel() == 0) {
    return emptyBinaryOp(self, other);
  }

  ideep::tensor& x = itensor_from_onednn(self);
  ideep::tensor& y = itensor_from_onednn(other);

  ideep::tensor z;
  const std::vector<float> scales{1.0, alpha.to<float>()};
  ideep::sum::compute(scales, {x, y}, z);

  return new_with_itensor_onednn(std::move(z), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor& onednn_add_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return native::onednn_add_out(self, other, alpha, self);
}

Tensor& onednn_mul_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_CHECK(result.sizes() == self.sizes(),
             "onednn_mul_out: the output size should be same as input size");
  ideep::tensor& z = itensor_from_onednn(result);
  ideep::tensor& x = itensor_from_onednn(self);

  // for zero_dim tensor
  if (other.ndimension() == 0) {
    ideep::eltwise_forward::compute(
      x, z, ideep::algorithm::eltwise_linear,
      ideep::prop_kind::forward_inference, /*alpha*/ other.item().to<float>());

    return result;
  } else {
    TORCH_CHECK(self.sizes() == other.sizes(),
               "onednn_mul_out: currently onednn not support broadcasting");
    ideep::tensor y = itensor_from_onednn(other);
    ideep::binary::compute(x, y, z, dnnl::algorithm::binary_mul);

    return result;
  }
}

Tensor onednn_mul(const Tensor& self, const Tensor& other) {
  if (self.numel() == 0 || other.numel() == 0) {
    return emptyBinaryOp(self, other);
  }
  Tensor result = empty_onednn(self.sizes(), optTypeMetaToScalarType(self.options().dtype_opt()),
                               self.options().layout_opt(), self.options().device_opt(),
                               self.options().pinned_memory_opt());
  return native::onednn_mul_out(self, other, result);
}

Tensor& onednn_mul_(Tensor& self, const Tensor& other) {
  return native::onednn_mul_out(self, other, self);
}

} // namespace native
} // namespace at

#endif // AT_ONEDNN_ENABLED
