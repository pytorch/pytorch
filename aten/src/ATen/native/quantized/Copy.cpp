#include <ATen/ATen.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/quantized/Copy.h>
#include <c10/core/MemoryFormat.h>
#include <c10/util/irange.h>

namespace at {
namespace native {

// Copying from float to QInt, used for assigning float value to QTensor
// The second exception condition `self.is_contiguous() && src.is_contiguous()`
// forces both the self & src tensors to be contiguous.
// This means that assignment of a non-contiguous quantized subtensor is currently not supported in pytorch
// e.g., Consider a 2x2 quantized tensor qt1 and a non-quantized tensor t2. The operation
// `qt1[:, 0] = t2[:, 0]` would trigger the exception b/c neither the LHS nor RHS is contiguous
Tensor& quantized_copy_from_float_(Tensor& self, const Tensor& src) {
  TORCH_CHECK(
      src.scalar_type() == at::kFloat,
      "Quantized copy only works with kFloat as source Tensor");
  TORCH_CHECK(
      (self.is_contiguous() && src.is_contiguous()) ||
      (self.is_contiguous(at::MemoryFormat::ChannelsLast) && src.is_contiguous(at::MemoryFormat::ChannelsLast)),
      "Quantized copy only works with contiguous and NHWC Tensors");
  TORCH_CHECK(
      self.sizes().equals(src.sizes()),
      "Quantized copy only works with Tensors with the same shape");
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "Copy", [&]() {
    if (self.qscheme() == kPerChannelAffine || self.qscheme() == kPerChannelAffineFloatQParams
        || self.qscheme() == kPerChannelSymmetric) {
      quantize_tensor_per_channel_affine(src, self, self.q_per_channel_scales(),
                                         self.q_per_channel_zero_points(),
                                         self.q_per_channel_axis());
    } else {
      quantize_tensor_per_tensor_affine(src, self, self.q_scale(), self.q_zero_point());
    }
  });
  return self;
}
} // namespace native
} // namespace at
