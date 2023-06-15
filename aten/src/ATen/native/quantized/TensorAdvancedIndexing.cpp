#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/quantized/IndexKernel.h>
#include <ATen/native/TensorAdvancedIndexingUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <c10/core/QScheme.h>
#include <ATen/native/TensorAdvancedIndexing.h>

namespace at {
namespace native {
DEFINE_DISPATCH(masked_fill_kernel_quantized_stub);
DEFINE_DISPATCH(index_put_kernel_quantized_stub);
DEFINE_DISPATCH(index_put_with_sort_quantized_stub);

namespace {
static TensorIterator make_index_put_iterator(const AdvancedIndex& info, const Tensor& value) {
  TORCH_CHECK(is_expandable_to(value.sizes(), info.src.sizes()), "shape mismatch: value tensor of shape ", value.sizes(),
             " cannot be broadcast to indexing result of shape ", info.src.sizes());
  TensorIteratorConfig config;
  // info.src is restrided by restride_src with 0 strided dimensions
  config.set_check_mem_overlap(false);
  config.resize_outputs(false);
  config.check_all_same_dtype(false);
  config.add_output(info.src);
  config.add_input(value);
  for (auto& index : info.indices) {
    config.add_input(index);
  }
  return config.build();
}

static Tensor & masked_fill_impl_quantized_cpu(Tensor & self, const Tensor & mask, const Scalar& value) {
  NoNamesGuard guard;
  TORCH_CHECK(mask.dtype() == ScalarType::Bool, "masked_fill only supports boolean masks, "
    "but got dtype ", mask.dtype());

  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
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

static Tensor & masked_fill_impl_quantized_cuda(Tensor& self, const Tensor & mask, const Scalar& value) {
  TORCH_CHECK(self.device() == mask.device(), "expected self and mask to be on the same device, but got mask on ",
    mask.device(), " and self on ", self.device());
  TORCH_CHECK(mask.scalar_type() == kBool, "masked_fill only supports boolean masks, "
    "but got dtype ", mask.scalar_type());
  TORCH_CHECK(self.qscheme() == c10::kPerTensorAffine, "masked_fill__quantized_cpu for quantized tensors is currently only supported for per tensor quantized tensors");

  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");

  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
      "Use of masked_fill_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }
  at::assert_no_partial_overlap(self, mask);

  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_fill_");

  auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(self)
      .add_input(self)
      .add_input(*b_mask)
      .build();

  masked_fill_kernel_quantized_stub(iter.device_type(), iter, value, self.q_scale(), self.q_zero_point());
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

Tensor & masked_fill__quantized_cuda(Tensor& self, const Tensor & mask, const Scalar& value) {
  TORCH_CHECK(!self.device().is_cpu(), "masked_fill_: Expected inputs to be on same device")
  return masked_fill_impl_quantized_cuda(self, mask, value);
}

Tensor & masked_fill__quantized_cuda(Tensor& self, const Tensor & mask, const Tensor & value) {
  TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", value.dim(), " dimension(s).");
  TORCH_CHECK(!self.device().is_cpu(), "masked_fill_: Expected inputs to be on same device")
  return masked_fill_impl_quantized_cuda(self, mask, value.item());
}

Tensor& _index_put_impl_quantized_cpu_(Tensor & self, const torch::List<c10::optional<Tensor>>& indices, const Tensor & value, const bool accumulate, const bool unsafe) {
  TORCH_CHECK_INDEX(indices.size() <= (size_t)self.dim(), "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  TORCH_CHECK(!value.is_quantized(), "Value argument for quantized input_put should not be quantized");
  TORCH_CHECK(self.qscheme() == c10::kPerTensorAffine, "index_put for quantized tensors is currently only supported for per tensor quantized tensors");
  TORCH_CHECK(!accumulate, "index_put for quantized tensors is currently only supported for accumulate=False");

  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
      "Use of index_put_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[indices] = tensor");
  }

  auto masked_fill_dispatch = canDispatchToMaskedFill(self, indices, value);
  if (std::get<0>(masked_fill_dispatch)) {
    return self.masked_fill_(std::get<1>(masked_fill_dispatch), value.item());
  }

  auto value_ = value;
  if (value.device() != self.device() && value.numel() == 1 && value.dim() == 0) {
    value_ = value.to(self.device());
  }
  at::assert_no_overlap(self, value);
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const c10::optional<Tensor>& index: indices) {
    if (index.has_value()) {
      at::assert_no_overlap(self, *index);
    }
  }

  auto info = make_info(self, indices);
  auto iter = make_index_put_iterator(info, value_);
  index_put_kernel_quantized_stub(iter.device_type(), iter, info.indexed_sizes, info.indexed_strides, accumulate, self.q_scale(), self.q_zero_point());
  return self;
}

Tensor& _index_put_impl_quantized_cuda_(Tensor & self, const torch::List<c10::optional<Tensor>>& indices, const Tensor & value, const bool accumulate, const bool unsafe) {
  TORCH_CHECK_INDEX(indices.size() <= (size_t)self.dim(), "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  TORCH_CHECK(!value.is_quantized(), "Value argument for quantized input_put should not be quantized");
  TORCH_CHECK(self.qscheme() == c10::kPerTensorAffine, "index_put for quantized tensors is currently only supported for per tensor quantized tensors");
  TORCH_CHECK(!accumulate, "index_put for quantized tensors is currently only supported for accumulate=False");

  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
      "Use of index_put_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[indices] = tensor");
  }

  auto masked_fill_dispatch = canDispatchToMaskedFill(self, indices, value);
  if (std::get<0>(masked_fill_dispatch)) {
    return self.masked_fill_(std::get<1>(masked_fill_dispatch), value.item());
  }

  auto value_ = value;
  if (value.device() != self.device() && value.numel() == 1 && value.dim() == 0) {
    value_ = value.to(self.device());
  }
  TORCH_CHECK(value.device() == self.device(), "expected device ", self.device(), " but got device ", value.device(), " for value tensor");

  at::assert_no_overlap(self, value);
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const c10::optional<Tensor>& index: indices) {
    if (index.has_value()) {
      at::assert_no_overlap(self, *index);
    }
  }

  // See Note [Enabling Deterministic Operations]
  if (self.device().type() == DeviceType::CUDA && globalContext().deterministicAlgorithms()) {
      index_put_with_sort_quantized_stub(self.device().type(), self, indices, value_, self.q_scale(), self.q_zero_point(), unsafe);
      return self;
  }

  auto info = make_info(self, indices);
  auto iter = make_index_put_iterator(info, value_);
  index_put_kernel_quantized_stub(iter.device_type(), iter, info.indexed_sizes, info.indexed_strides, accumulate, self.q_scale(), self.q_zero_point());
  return self;
}

}
}
