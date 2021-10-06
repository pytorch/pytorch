#include <ATen/native/quantized/Copy.h>

#include <ATen/ATen.h>
#include <ATen/native/quantized/affine_quantizer.h>
#include <ATen/quantized/Quantizer.h>

static inline void run_checks_and_set_quantizer_(at::Tensor& self, const at::Tensor& src) {
  if (self.is_quantized() && src.is_quantized()) {
    TORCH_CHECK(self.qscheme() == src.qscheme(),
                "Quantized Copy only works with same qscheme");
    TORCH_CHECK(self.scalar_type() == src.scalar_type());
    at::set_quantizer_(self, src.quantizer());
  }
  if (!self.is_quantized() && src.is_quantized()) {
    TORCH_CHECK(false, "Copying from quantized Tensor to non-quantized Tensor is not allowed, please use dequantize to get a float Tensor from a quantized Tensor");
  }
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

Tensor& copy_quantized_cpu_(Tensor& self, const Tensor& src, bool non_blocking) {
  if (self.is_quantized() && !src.is_quantized()) {
    return quantized_copy_from_float_cpu_(self, src);
  }
  run_checks_and_set_quantizer_(self, src);
  c10::impl::ExcludeDispatchKeyGuard guard_(DispatchKey::QuantizedCPU);
  c10::impl::IncludeDispatchKeyGuard inc_guard_(DispatchKey::CPU);
  return self.copy_(src, non_blocking); // redispatch!
}

Tensor& copy_quantized_cuda_(Tensor& self, const Tensor& src, bool non_blocking) {
  run_checks_and_set_quantizer_(self, src);
  c10::impl::ExcludeDispatchKeyGuard guard_(DispatchKey::QuantizedCUDA);
  c10::impl::IncludeDispatchKeyGuard inc_guard_(DispatchKey::CUDA);
  return self.copy_(src, non_blocking); // redispatch!
}

Tensor& copy_quantized_xpu_(Tensor& self, const Tensor& src, bool non_blocking) {
  run_checks_and_set_quantizer_(self, src);
  c10::impl::ExcludeDispatchKeyGuard guard_(DispatchKey::QuantizedXPU);
  c10::impl::IncludeDispatchKeyGuard inc_guard_(DispatchKey::XPU);
  return self.copy_(src, non_blocking); // redispatch!
}
} // namespace native
} // namespace at
