#include <ATen/native/quantized/Copy.h>

#include <ATen/ATen.h>
#include <ATen/native/quantized/affine_quantizer.h>
#include <ATen/quantized/Quantizer.h>

static inline at::Tensor& copy_quantized_(
    at::DispatchKey exclude_key,
    at::DispatchKey include_key,
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  TORCH_CHECK(
      self.is_quantized() && src.is_quantized(),
      "Both tensors should be quantized")
  TORCH_CHECK(
      self.qscheme() == src.qscheme(),
      "Quantized Copy only works with same qscheme");
  TORCH_CHECK(self.scalar_type() == src.scalar_type());
  set_quantizer_(self, src.quantizer());

  c10::impl::ExcludeDispatchKeyGuard exclude_guard(exclude_key);
  c10::impl::IncludeDispatchKeyGuard include_guard(include_key);
  return self.copy_(src, non_blocking); // redispatch!
}
namespace at {
namespace native {

// Copying from float to QInt, used for assigning float value to QTensor
Tensor& quantized_copy_from_float_cpu_(Tensor& self, const Tensor& src) {
  TORCH_CHECK(
      src.scalar_type() == at::kFloat,
      "Quantized copy only works with kFloat as source Tensor");
  TORCH_CHECK(
      self.is_contiguous() && src.is_contiguous(),
      "Quantized copy only works with contiguous Tensors");
  TORCH_CHECK(
      self.sizes().equals(src.sizes()),
      "Quantized copy only works with Tensors with the same shape");
  TORCH_CHECK(
      self.device().type() == kCPU,
      "Quantized copy only works with QuantizedCPU Tensors");
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "Copy", [&]() {
    float* src_data = src.data_ptr<float>();
    scalar_t* self_data = self.data_ptr<scalar_t>();
    for (int i = 0; i < self.numel(); ++i) {
      self_data[i] = quantize_val<scalar_t>(
          self.q_scale(), self.q_zero_point(), src_data[i]);
    }
  });
  return self;
}

Tensor& copy_quantized_cpu_(
    Tensor& self,
    const Tensor& src,
    bool non_blocking) {
  TORCH_CHECK(
    !self.is_quantized() && src.is_quantized(),
    "Copying from quantized Tensor to non-quantized Tensor is not allowed, please use dequantize to get a float Tensor from a quantized Tensor"
  );
  if (self.is_quantized() && !src.is_quantized()) {
    return quantized_copy_from_float_cpu_(self, src);
  }
  return copy_quantized_(
      DispatchKey::QuantizedCPU, DispatchKey::CPU, self, src, non_blocking);
}

Tensor& copy_quantized_cuda_(
    Tensor& self,
    const Tensor& src,
    bool non_blocking) {
  return copy_quantized_(
      DispatchKey::QuantizedCUDA, DispatchKey::CUDA, self, src, non_blocking);
}

Tensor& copy_quantized_xpu_(
    Tensor& self,
    const Tensor& src,
    bool non_blocking) {
  return copy_quantized_(
      DispatchKey::QuantizedXPU, DispatchKey::XPU, self, src, non_blocking);
}

} // namespace native
} // namespace at
