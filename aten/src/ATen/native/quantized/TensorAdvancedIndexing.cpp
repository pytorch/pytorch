#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/quantized/IndexKernel.h>
#include <ATen/native/TensorAdvancedIndexingUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <c10/core/QScheme.h>

namespace at {
namespace native {
DEFINE_DISPATCH(masked_fill_kernel_quantized_stub);

namespace {
static Tensor & masked_fill_impl_quantized_cpu(Tensor & self, const Tensor & mask, const Scalar& value) {
  NoNamesGuard guard;
  if (mask.dtype() == ScalarType::Byte) {
    TORCH_WARN("masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
  }

  if (at::has_internal_overlap(self) == MemOverlap::YES) {
    TORCH_WARN(
      "Use of masked_fill_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }
  at::assert_no_partial_overlap(self, mask);

  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)  // deprecated, but not a hard error
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .add_output(self)
    .add_input(mask)
    .build();

  masked_fill_kernel_quantized_stub(iter.device_type(), iter, value, self.q_scale(), self.q_zero_point());
  return self;
}
}

Tensor & masked_fill__quantized_cpu(Tensor& self, const Tensor & mask, const Scalar& value) {
  TORCH_CHECK(self.qscheme() == c10::kPerTensorAffine, "masked_fill__quantized_cpu for quantized tensors is currently only supported for per tensor quantized tensors");
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");

  masked_fill_impl_quantized_cpu(self, mask, value);
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

Tensor & masked_fill__quantized_cpu(Tensor& self, const Tensor & mask, const Tensor & value) {
  TORCH_CHECK(self.qscheme() == c10::kPerTensorAffine, "masked_fill__quantized_cpu for quantized tensors is currently only supported for per tensor quantized tensors");
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
  TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", value.dim(), " dimension(s).");

  masked_fill_impl_quantized_cpu(self, mask, value.item());
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

}
}
