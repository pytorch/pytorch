#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/IndexKernel.h>
#include <ATen/native/TensorAdvancedIndexing.h>  // For at::native::index_out
#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/CUDAFunctions.h>
#else
#include <ATen/ops/index_cuda_dispatch.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/masked_scatter_native.h>
#include <ATen/ops/masked_select_native.h>
#endif


namespace at::native {

static Tensor & masked_select_out_cuda_impl(Tensor & result, const Tensor & self, const Tensor & mask) {
  NoNamesGuard guard;

  TORCH_CHECK(mask.scalar_type() == ScalarType::Bool,
              "masked_select: expected BoolTensor for mask");
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
              "masked_select(): self and result must have the same scalar type");

  auto mask_temp = (mask.dim() == 0)
    ? c10::MaybeOwned<Tensor>::owned(mask.unsqueeze(0))
    : c10::MaybeOwned<Tensor>::borrowed(mask);
  auto self_temp = (self.dim() == 0)
    ? c10::MaybeOwned<Tensor>::owned(self.unsqueeze(0))
    : c10::MaybeOwned<Tensor>::borrowed(self);

  // Cannot reassign to mask_temp and self_temp here! if they are
  // owning and expand_outplace returns a borrow, the returned borrow
  // would dangle.
  auto [mask_expanded, self_expanded] = expand_outplace(*mask_temp, *self_temp);
  at::cuda::index_out(
      result, *self_expanded,
      c10::List<std::optional<at::Tensor>>({*std::move(mask_expanded)}));

  return result;
}

Tensor masked_select_cuda(const Tensor & self, const Tensor & mask) {
  namedinference::compute_broadcast_outnames(self, mask);
  Tensor result = at::empty({0}, self.options());
  return masked_select_out_cuda_impl(result, self, mask);
}

Tensor & masked_select_out_cuda(const Tensor & self, const Tensor & mask, Tensor & result) {
  namedinference::compute_broadcast_outnames(self, mask);
  return masked_select_out_cuda_impl(result, self, mask);
}

Tensor & masked_scatter__cuda(Tensor& self, const Tensor& mask, const Tensor& source) {
  at::assert_no_internal_overlap(self);
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      "masked_scatter_: expected self and source to have same dtypes but got ",
      self.scalar_type(),
      " and ",
      source.scalar_type());
  TORCH_CHECK(mask.dtype() == ScalarType::Bool, "masked_scatter_ only supports boolean masks, "
     "but got mask with dtype ", mask.dtype());

  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_scatter_");

  if (self.numel() == 0) {
    return self;
  }

  auto maskPrefixSum = at::empty(self.sizes(), mask.options().dtype(kLong));
  launch_masked_scatter_kernel(self, *b_mask, maskPrefixSum, source);

  return self;
}

}  // namespace at::native
