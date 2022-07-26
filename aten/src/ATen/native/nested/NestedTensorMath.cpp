#include <ATen/native/nested/NestedTensorMath.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/layer_norm.h>
#include <ATen/NestedTensorImpl.h>
#include <c10/core/DispatchKey.h>
#include <ATen/native/nested/NestedTensorMath.h>

namespace at {
namespace native {

namespace {
template <typename Func>
Tensor map_nt(const Tensor& nt, Func f) {
  auto* nt_impl = get_nested_tensor_impl(nt);
  const auto& sizes = nt_impl->get_nested_size_tensor();
  return at::detail::make_tensor<NestedTensorImpl>(f(nt_impl->get_buffer()), sizes);
}

c10::optional<int64_t> maybe_get_consistent_last_dim_of_nested_tensor(
    const NestedTensorImpl& nt) {
  const auto& sizes = nt.get_nested_size_tensor();
  // The last entry in every row of sizes must be the same.
  const auto& last_dims = sizes.select(1, -1);
  const auto last_dims_accessor = last_dims.packed_accessor64<int64_t, 1>();
  // REVIEW: this can't be the most efficient and concise way to
  // write this check, can it?
  const auto last_dim_value = last_dims_accessor[0];
  for (const auto i : c10::irange(1, last_dims.numel())) {
    if (last_dims_accessor[i] != last_dim_value) {
      return c10::nullopt;
    }
  }
  return last_dim_value;
}

int64_t num_bytes(IntArrayRef sizes) {
  // 0-dim Tensors have torch.Size of .size() 0, but carry 1 memory.
  // Empty 1-dim Tensors (torch.tensor([])) have torch.Size of .size() 1,
  // but carry 0 memory.
  int64_t result = 1;
  int64_t stride = 1;
  for (int ii = sizes.size() - 1; ii >= 0; --ii) {
    result += (sizes[ii] - 1) * stride;
    // TODO: accept strides as input when we support them instead of
    // assuming contiguous.
    stride *= sizes[ii];
  }
  return result;
}

std::vector<int64_t> NestedTensor_get_max_size_from_size_tensor(const Tensor& sizes) {
  if (sizes.dim() == 0) {
    return {};
  }
  const auto sizes_ptr = sizes.data_ptr<int64_t>();
  const auto sizes_size_0 = sizes.sizes()[0];
  const auto sizes_size_1 = sizes.sizes()[1];
  TORCH_INTERNAL_ASSERT(sizes_size_1 > 0);
  std::vector<int64_t> results(sizes_size_1, 0);
  for (const auto ii : c10::irange(sizes_size_0)) {
    for (const auto jj : c10::irange(sizes_size_1)) {
      auto val = sizes_ptr[ii * sizes_size_1 + jj];
      if (results[jj] < val) {
        results[jj] = val;
      }
    }
  }
  return results;
}

Tensor pad_tensor_to_shape(
    const Tensor& t,
    IntArrayRef goal_shape,
    double value = 0) {
  std::vector<int64_t> padd;
  auto tup = t.sizes();
  TORCH_CHECK(
      t.dim() == (int64_t)(goal_shape.size()),
      "dimension ",
      t.dim(),
      " doesn't match length ",
      goal_shape.size(),
      " of goal shape.");
  for (int64_t i = tup.size() - 1; i >= 0; i--) {
    padd.push_back(0);
    padd.push_back(goal_shape[i] - tup[i]);
  }
  Tensor new_tensor = at::constant_pad_nd(t, IntArrayRef(padd), value);
  new_tensor = new_tensor.reshape(goal_shape);
  return new_tensor;
}
} // namespace

at::Tensor wrap_buffer(at::Tensor buffer, at::Tensor nested_size_tensor) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(buffer.is_contiguous(), "Given buffer must be contiguous.");
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(buffer), std::move(nested_size_tensor));
}

at::Tensor wrap_buffer(
    at::Tensor buffer, at::Tensor nested_size_tensor,
    at::Tensor nested_stride_tensor, const std::vector<int64_t>& offsets) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(buffer.is_contiguous(), "Given buffer must be contiguous.");
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(buffer), std::move(nested_size_tensor),
      std::move(nested_stride_tensor), offsets);
}

std::vector<at::Tensor> NestedTensor_unbind(
    const at::Tensor& self,
    int64_t dim) {
  TORCH_CHECK(
      dim == 0,
      "NestedTensor can only be unbound along dimension 0 ",
      "got dimension ",
      dim,
      " instead.");
  auto self_ptr = get_nested_tensor_impl(self);
  int64_t ntensors = self_ptr->size(0);
  std::vector<at::Tensor> result_tensors(ntensors);
  if (ntensors == 0) {
    return result_tensors;
  }
  const at::Tensor& buffer = self_ptr->get_buffer();
  std::vector<IntArrayRef> sizes = NestedTensor_get_sizes(self_ptr),
      strides = NestedTensor_get_strides(self_ptr);
  const std::vector<int64_t>& offsets = self_ptr->get_offsets();
  for (int64_t i = 0; i < ntensors; i++) {
    result_tensors[i] = buffer.as_strided(sizes[i], strides[i], offsets[i]);
  }
  return result_tensors;
}

Tensor& NestedTensor_relu_(Tensor& self) {
  at::relu_(const_cast<Tensor&>(get_nested_tensor_impl(self)->get_buffer()));
  return self;
}

Tensor NestedTensor_relu(const Tensor& self) {
  return map_nt(self, at::relu);
}

Tensor& NestedTensor_gelu_(Tensor& self, c10::string_view approximate) {
  at::gelu_(const_cast<Tensor&>(get_nested_tensor_impl(self)->get_buffer()), approximate);
  return self;
}

Tensor NestedTensor_gelu(const Tensor& self, c10::string_view approximate) {
  return map_nt(
      self,
      [approximate](const Tensor& buffer) {
        return at::gelu(buffer, approximate);
      });
}

Tensor NestedTensor_nested_tensor_from_mask(const Tensor& t, const Tensor& mask) {
    TORCH_CHECK(mask.scalar_type() == at::ScalarType::Bool, "Expected mask to be of ScalarType Bool, but got ", mask.scalar_type(), " instead.");
    TORCH_CHECK(mask.dim() == 2, "Padding mask should be 2D");
    TORCH_CHECK(t.dim() == 3, "Input should be a 3D tensor, N * L * D");
    auto N = t.size(0), L = t.size(1), D = t.size(2);
    auto NN = mask.size(0), LL = mask.size(1);
    TORCH_CHECK(N == NN && L == LL, "Mask size should match input size");

    // N * L
    Tensor sizes = mask;
    Tensor tmp_pad = at::zeros({N, 1}, mask.options());
    // Make sure padding is only added at the end of mask
    Tensor nums = at::cat({sizes, tmp_pad}, 1).to(kInt).argmin(1);

    // N, ([size1, size2, ... sizeN])
    sizes = sizes.cumsum(1).select(1, L - 1);
    nums = nums.to(sizes.options());

    TORCH_CHECK(sizes.equal(nums), "Mask must be left-aligned without gaps");

    sizes = sizes.reshape({N, 1});
    // N, ([d1=D, d2=D, ... dN=D])
    Tensor d = at::full_like(sizes, D);

    // N * 2, ([[size1, D], [size2, D], ..., [sizeN, D]])
    sizes = at::cat({sizes, d}, 1).to(kCPU);

    return at::_nested_from_padded(t, sizes, false);
}

bool NestedTensor_nested_tensor_from_mask_left_aligned(const Tensor& t, const Tensor& mask) {
    TORCH_CHECK(mask.scalar_type() == at::ScalarType::Bool, "Expected mask to be of ScalarType Bool, but got ", mask.scalar_type(), " instead.");
    TORCH_CHECK(mask.dim() == 2, "Padding mask should be 2D");
    TORCH_CHECK(t.dim() == 3, "Input should be a 3D tensor, N * L * D");
    auto N = t.size(0), L = t.size(1);
    auto NN = mask.size(0), LL = mask.size(1);
    TORCH_CHECK(N == NN && L == LL, "Mask size should match input size");

    // N * L
    Tensor sizes = mask;
    Tensor tmp_pad = at::zeros({N, 1}, mask.options());
    // Make sure padding is only added at the end of mask
    Tensor nums = at::cat({sizes, tmp_pad}, 1).to(kInt).argmin(1);

    // N, ([size1, size2, ... sizeN])
    sizes = sizes.cumsum(1).select(1, L - 1);
    nums = nums.to(sizes.options());

    return sizes.equal(nums);
}

Tensor nested_tensor(
    TensorList list,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  TensorOptions options_ =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);

  if (list.size() == 0) {
    return wrap_buffer(ones({0}, dtype, layout, device), ones({}));
  }
  std::vector<Tensor> sizes;
  std::vector<Tensor> flat_tensors;
  for (const auto i : c10::irange(list.size())) {
    if (i > 0) {
      int64_t dim_i = list[i].dim();
      int64_t dim_prev = list[i - 1].dim();
      TORCH_CHECK(
          dim_i == dim_prev,
          "All Tensors given to nested_tensor must have the same dimension. ",
          "Found dimension ",
          dim_i,
          " for Tensor at index ",
          i,
          " and dimension ",
          dim_prev,
          " for Tensor at index ",
          i - 1,
          ".");
    }
    // TODO: Remove call to contiguous once we support strides.
    flat_tensors.push_back(list[i].reshape(-1).contiguous());
    sizes.push_back(tensor(c10::IntArrayRef(list[i].sizes())));
  }

  TensorOptions options = flat_tensors[0].options().merge_in(options_);

  return wrap_buffer(
      at::cat(flat_tensors).to(options), at::native::stack(sizes));
}

int64_t get_consistent_last_dim_of_nested_tensor(const NestedTensorImpl& nt) {
  auto result = maybe_get_consistent_last_dim_of_nested_tensor(nt);
  TORCH_CHECK(
      result.has_value(),
      "all tensors in NestedTensor must have the same trailing dim for Matmul but got ",
      nt.get_nested_size_tensor().select(1, -1));
  return *result;
}

std::vector<int64_t> NestedTensor_get_max_size(const NestedTensorImpl& nt) {
  return NestedTensor_get_max_size_from_size_tensor(nt.get_nested_size_tensor());
}

Tensor NestedTensor_layer_norm(
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    double eps) {
  TORCH_CHECK(weight_opt && bias_opt, "NestedTensor layer_norm requires weight and bias");
  const auto& weight = *weight_opt;
  const auto& bias = *bias_opt;
  TORCH_CHECK(!weight.is_nested(), "NestedTensor weight not supported for layer_norm");
  TORCH_CHECK(!bias.is_nested(), "NestedTensor bias not supported for layer_norm");
  auto* nt_input = get_nested_tensor_impl(input);
  TORCH_CHECK(nested_tensor_impl_is_contiguous(nt_input));
  const auto& input_buffer = nt_input->get_buffer();
  const auto last_dim = get_consistent_last_dim_of_nested_tensor(*nt_input);
  const auto valid_word_num = input_buffer.numel() / last_dim;
  const auto weight_contig = weight.expect_contiguous();
  const auto bias_contig = bias.expect_contiguous();
  auto output_buffer = at::native::empty_like(
      input_buffer,
      c10::nullopt /* dtype */,
      c10::nullopt /* layout */,
      c10::nullopt /* device */,
      c10::nullopt /* pin_memory */,
      at::MemoryFormat::Contiguous);
  auto options = input_buffer.options();
  if (input_buffer.is_cuda()) {
    auto acc_type = at::toAccumulateType(input_buffer.scalar_type(), true);
    options = options.dtype(acc_type);
  }
  Tensor mean = at::empty({valid_word_num}, options);
  Tensor rstd = at::empty({valid_word_num}, options);
  LayerNormKernel(
      input_buffer.is_cuda() ? kCUDA : kCPU,
      input_buffer,
      *weight_contig,
      *bias_contig,
      valid_word_num,
      last_dim,
      eps,
      &output_buffer,
      &mean,
      &rstd);
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(output_buffer), nt_input->get_nested_size_tensor());
}

Tensor NestedTensor_from_padded_and_nested_example(
    const Tensor& padded,
    const Tensor& nt_example) {
  return _nested_from_padded(padded, get_nested_tensor_impl(nt_example)->get_nested_size_tensor());
}

Tensor nested_from_padded_generic(
    const Tensor& padded,
    const Tensor& sizes,
    const bool do_transform_0213) {
  // Check and do transform 0213
  auto padded_transformed = padded;
  if (do_transform_0213) {
    padded_transformed = padded.permute({0, 2, 1, 3})
      .contiguous()
      .view(
          {padded.size(0),
           padded.size(2),
           padded.size(1) * padded.size(3)});
  }
  auto target_size = NestedTensor_get_max_size_from_size_tensor(sizes);
  // There may be extra padding on padded beyond the max size in the nested tensor.
  // Make the mask size match.
  const size_t dim = padded_transformed.dim();
  TORCH_CHECK(dim - 1 == target_size.size(), "dim: ", dim, "target_size: ", target_size.size());
  for (size_t ii = 0; ii < dim - 1; ++ii) {
    const auto padded_size_i = padded_transformed.sizes()[ii + 1];
    if (target_size[ii] < padded_size_i) {
      target_size[ii] = padded_size_i;
    }
  }
  IntArrayRef target_size_arr(target_size);
  std::vector<at::Tensor> masks;
  std::vector<at::Tensor> all_sizes = sizes.unbind();
  for (const auto& size : all_sizes) {
    IntArrayRef sizes_i(
        size.data_ptr<int64_t>(), size.data_ptr<int64_t>() + size.numel());
    at::Tensor mask_i = padded_transformed.new_full(
        sizes_i, true, kBool, c10::nullopt, c10::nullopt, c10::nullopt);
    masks.push_back(pad_tensor_to_shape(mask_i, target_size_arr));
  }
  at::Tensor final_mask = at::stack(masks);
  at::Tensor new_buffer = padded_transformed.masked_select(final_mask).to(padded.device());
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(new_buffer), sizes);
}

Tensor NestedTensor_to_padded_tensor_generic(
    const Tensor& t,
    double padding,
    OptionalIntArrayRef output_size) {
  // TODO: support noncontiguous case
  // error out for now
  TORCH_CHECK(
      nested_tensor_impl_is_contiguous(get_nested_tensor_impl(t)),
      "for now to_padded_tensor only supports contiguous nested tensor");
  // TODO: skipped optimization for case of all 1x1 tensors
  auto& nt = *get_nested_tensor_impl(t);
  auto max_size = NestedTensor_get_max_size(nt);
  auto sizes = nt.get_nested_size_tensor();

  if (sizes.numel() == 0 || sizes.dim() == 0) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(nt.get_buffer().numel() == 0);
    return nt.get_buffer().clone();
  }

  // TODO: doesn't handle empty/scalar entries because we don't need
  // it for transformers; see to_padded_tensor in
  // pytorch/nestedtensor's masking.cpp.

  const auto sizes_num_rows = sizes.sizes()[0];
  const auto sizes_num_columns = sizes.sizes()[1];
  const auto sizes_data_start = sizes.data_ptr<int64_t>();
  const auto sizes_data_end = sizes_data_start + sizes.numel();
  std::vector<int64_t> split_sizes;
  split_sizes.reserve(sizes_num_rows);
  for (auto sizes_data = sizes_data_start; sizes_data != sizes_data_end;
       sizes_data += sizes_num_columns) {
    split_sizes.push_back(
        num_bytes(IntArrayRef(sizes_data, sizes_num_columns)));
  }
  std::vector<int64_t> nonzero_split_sizes;
  for (const auto split_size : split_sizes) {
    if (split_size > 0) {
      nonzero_split_sizes.push_back(split_size);
    }
  }
  const auto buffer = nt.get_buffer();
  std::vector<Tensor> buffers_;
  if (!nonzero_split_sizes.empty()) {
    buffers_ = at::split_with_sizes(buffer, nonzero_split_sizes, 0);
  }

  std::vector<Tensor> buffers;
  buffers.reserve(split_sizes.size());
  int64_t next_buffer = 0;
  auto sizes_ptr = sizes_data_start;
  for (const auto split_size : split_sizes) {
    Tensor to_pad;
    IntArrayRef tensor_sizes(sizes_ptr, sizes_num_columns);
    if (split_size > 0) {
      to_pad = buffers_[next_buffer++].reshape(tensor_sizes);
    } else {
      to_pad = at::empty(tensor_sizes, buffer.options());
    }
    buffers.push_back(pad_tensor_to_shape(to_pad, max_size, padding));
    sizes_ptr += sizes_num_columns;
  }
  auto ret_val = at::stack(buffers);

  // Pad output tensor to output_size if provided
  if (output_size.has_value()) {
    auto output_size_ = output_size.value();
    TORCH_CHECK(
        (int64_t)output_size_.size() == ret_val.dim(),
        "Length of output_size does not match NestedTensor dims. Broadcasting is not supported.");
    for (int64_t i = 0; i < (int64_t)ret_val.dim(); i++) {
      TORCH_CHECK(
          output_size_[i] >= ret_val.size(i),
          "Value in output_size is less than NestedTensor padded size. Truncation is not supported.");
    }
    return pad_tensor_to_shape(ret_val, output_size_, padding);
  }
  return ret_val;
}

Tensor NestedTensor_embedding(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  const auto* nt_indices = get_nested_tensor_impl(indices);
  TORCH_CHECK(
      !weight.is_nested(), "NestedTensor weight not supported for embedding");
  TORCH_CHECK(indices.dim() < 3);
  TORCH_CHECK(indices.dim() > 0, "NestedTensor embedding doesn't support empty indices.")
  TORCH_CHECK(weight.dim() == 2);
  TORCH_CHECK(nested_tensor_impl_is_contiguous(nt_indices));
  TORCH_CHECK(weight.is_contiguous());

  const auto& indices_buffer = nt_indices->get_buffer();
  auto result_buffer = at::embedding(
      weight, indices_buffer, padding_idx, scale_grad_by_freq, sparse);
  const auto& sizes = nt_indices->get_nested_size_tensor();
  auto new_sizes = at::empty({sizes.size(0)}, sizes.options());
  new_sizes.fill_(weight.sizes()[1]);
  new_sizes = new_sizes.reshape({new_sizes.size(0), 1});
  new_sizes = at::cat({sizes, new_sizes}, 1);
  return at::detail::make_tensor<NestedTensorImpl>(
      result_buffer.reshape({-1}), std::move(new_sizes));
}

std::pair<NestedTensorImpl*, NestedTensorImpl*>
get_elementwise_nested_tensor_impl(
    const Tensor& self,
    const Tensor& other,
    const std::string& op_name) {
  if (self.is_nested() && !(other.is_nested())) {
    TORCH_CHECK(
        false,
        "Expected both self and other to be nested, but got a nested self and non-nested other");
  } else if (!(self.is_nested()) && other.is_nested()) {
    TORCH_CHECK(
        false,
        "Expected both self and other to be nested, but got a non-nested self and nested other");
  } else if (!(self.is_nested()) || !(other.is_nested())) {
    TORCH_CHECK(
        false,
        "Expected both self and other to be nested, but got a non-nested self and non-nested other");
  }

  auto self_ptr = get_nested_tensor_impl(self);
  auto other_ptr = get_nested_tensor_impl(other);

  TORCH_CHECK(
      self.dim() == other.dim(),
      op_name,
      " does not support broadcasting when given a NestedTensor");
  TORCH_CHECK(
      at::equal(
          self_ptr->get_nested_size_tensor(),
          other_ptr->get_nested_size_tensor()),
      op_name,
      " does not support broadcasting when given a NestedTensor");
  TORCH_CHECK(
      nested_tensor_impl_is_contiguous(self_ptr) &&
          nested_tensor_impl_is_contiguous(other_ptr),
      op_name,
      " does not support non-contiguous NestedTensor inputs");
  return std::make_pair(self_ptr, other_ptr);
}

template <typename Func>
Tensor NestedTensor_elementwise_Tensor(
    const Tensor& self,
    const Tensor& other,
    const std::string& op_name,
    Func f) {
  // self is a scalar
  if (!self.is_nested() && self.dim() == 0 && self.numel() == 1) {
    auto other_impl = get_nested_tensor_impl(other);
    return wrap_buffer(
      f(self, other_impl->get_buffer()),
      other_impl->get_nested_size_tensor().clone()
    );
  }
  // other is a scalar
  if (!other.is_nested() && other.dim() == 0 && other.numel() == 1) {
    auto self_impl = get_nested_tensor_impl(self);
    return wrap_buffer(
      f(self_impl->get_buffer(), other),
      self_impl->get_nested_size_tensor().clone()
    );
  }
  NestedTensorImpl* self_impl = nullptr;
  NestedTensorImpl* other_impl = nullptr;
  std::tie(self_impl, other_impl) =
      get_elementwise_nested_tensor_impl(self, other, op_name);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self_impl);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(other_impl);
  const auto& nt_self = *self_impl;
  const auto& nt_other = *other_impl;
  const auto& self_sizes = nt_self.get_nested_size_tensor();
  return wrap_buffer(
      f(nt_self.get_buffer().reshape({-1}),
        nt_other.get_buffer().reshape({-1})),
      self_sizes);
}

Tensor NestedTensor_add_Tensor(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  return NestedTensor_elementwise_Tensor(
      self, other, "add", [alpha](const Tensor& b1, const Tensor& b2) {
        return at::add(b1, b2, alpha);
      });
}

Tensor NestedTensor_mul_Tensor(const Tensor& self, const Tensor& other) {
  return NestedTensor_elementwise_Tensor(
      self, other, "mul", [](const Tensor& b1, const Tensor& b2) {
        return at::mul(b1, b2);
      });
}

template <typename Func>
Tensor& NestedTensor_elementwise__Tensor(
    Tensor& self,
    const Tensor& other,
    const std::string& op_name,
    Func f) {
  // self is a scalar
  if (!self.is_nested() && self.dim() == 0 && self.numel() == 1) {
    auto other_impl = get_nested_tensor_impl(other);
    f(self, other_impl->get_buffer());
    return self;
  }
  // other is a scalar
  if (!other.is_nested() && other.dim() == 0 && other.numel() == 1) {
    auto self_impl = get_nested_tensor_impl(self);
    f(self_impl->get_buffer(), other);
    return self;
  }
  NestedTensorImpl* self_impl = nullptr;
  NestedTensorImpl* other_impl = nullptr;
  std::tie(self_impl, other_impl) =
      get_elementwise_nested_tensor_impl(self, other, op_name);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self_impl);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(other_impl);
  const auto& nt_self = *self_impl;
  const auto& nt_other = *other_impl;
  f(nt_self.get_buffer().view({-1}), nt_other.get_buffer().view({-1}));
  return self;
}

Tensor& NestedTensor_add__Tensor(
    Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  return NestedTensor_elementwise__Tensor(
      self, other, "add_", [alpha](const Tensor& b1, const Tensor& b2) {
        return b1.add_(b2, alpha);
      });
}

Tensor& NestedTensor_mul__Tensor(Tensor& self, const Tensor& other) {
  return NestedTensor_elementwise__Tensor(
      self, other, "mul_", [](const Tensor& b1, const Tensor& b2) {
        return b1.mul_(b2);
      });
}

Tensor select_nested(const Tensor& self, int64_t dim, int64_t index) {
  auto self_ptr = get_nested_tensor_impl(self);
  int64_t positive_dim = at::maybe_wrap_dim(dim, self_ptr->dim());
  TORCH_CHECK(
    positive_dim == 0,
    "NestedTensor can only be selected along dimension 0 ",
    "got dimension ", dim, " instead."
  );
  int64_t ntensors = self_ptr->size(0);
  TORCH_CHECK_INDEX(
      index >= -ntensors && index < ntensors,
      "index ", index,
      " is out of bounds for dimension 0 with size ", ntensors);
  int64_t positive_index = index < 0 ? index + ntensors : index;
  const at::Tensor& buffer = self_ptr->get_buffer();
  std::vector<IntArrayRef> sizes = NestedTensor_get_sizes(self_ptr),
      strides = NestedTensor_get_strides(self_ptr);
  const std::vector<int64_t>& offsets = self_ptr->get_offsets();
  return buffer.as_strided(sizes[positive_index], strides[positive_index], offsets[positive_index]);
}

Tensor clone_nested(
    const Tensor& self,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);
  TORCH_CHECK(
      memory_format == MemoryFormat::Preserve,
      "clone_nested only supports memory format Preserve, but got ",
      memory_format,
      " instead.");
  // TODO: The size doesn't necessarily need to be cloned, but it is more
  // conservative. This is something we could revisit once we land a more
  // efficient implementation of nested_size_tensor_.
  return wrap_buffer(
      get_buffer(self).clone(), get_nested_size_tensor(self).clone());
}

at::Tensor NestedTensor_get_nested_size_tensor(const at::Tensor& self){
  return get_nested_size_tensor(self);
}

Tensor dropout_nested(const Tensor& input, double p, bool train) {
  auto input_ptr = get_nested_tensor_impl(input);
  const Tensor& input_buffer = input_ptr->get_buffer(),
      & sizemat = input_ptr->get_nested_size_tensor(),
      & stridemat = input_ptr->get_nested_stride_tensor();
  const std::vector<int64_t>& offsets = input_ptr->get_offsets();
  Tensor output_buffer = at::dropout(input_buffer, p, train);
  // regular tensor dropout reuses input size and stride
  // i.e. if input is not contiguous, then output is also discontiguous
  return wrap_buffer(output_buffer, sizemat.clone(), stridemat.clone(), offsets);
}

Tensor& dropout_nested_(Tensor& input, double p, bool train) {
  Tensor input_buffer = get_buffer(input);
  at::dropout_(input_buffer, p, train);
  return input;
}

Tensor softmax_nested(const Tensor& input, const int64_t dim, const bool half_to_float) {
  auto input_ptr = get_nested_tensor_impl(input);
  int64_t ntensors = input_ptr->size(0);
  if (ntensors == 0) {
    return input.clone();
  }
  int64_t positive_dim = at::maybe_wrap_dim(dim, input_ptr->dim());
  TORCH_CHECK(
      positive_dim >= 1,
      "Cannot apply softmax across nested dimension 0");
  // create a contiguous output
  const Tensor& buffer = input_ptr->get_buffer(),
      & sizemat = input_ptr->get_nested_size_tensor();
  Tensor output_buffer = buffer.new_empty(buffer.sizes());
  Tensor output = wrap_buffer(output_buffer, sizemat.clone());
  // call tensor softmax
  // TODO: for cpu, maybe use `parallel_for` if benchmarks show necessity
  //       to do that, have to merge `aten/src/ATen/native/cpu/SoftMaxKernel.cpp/softmax_kernel`
  //       1. it has `parallel_for` and we cannot multi-thread in multi-thread
  //       2. cannot dispatch in multi-thread (in this case at::_softmax_out)
  std::vector<Tensor> input_unbind = input.unbind(),
      output_unbind = output.unbind();
  for (int64_t i = 0; i < ntensors; i++) {
    at::_softmax_out(
        output_unbind[i],
        input_unbind[i],
        positive_dim - 1,
        half_to_float);
  }
  return output;
}

Tensor bmm_nested(const Tensor& self, const Tensor& mat2) {
  auto self_ptr = get_nested_tensor_impl(self),
      mat2_ptr = get_nested_tensor_impl(mat2);
  TORCH_CHECK(self_ptr->dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(mat2_ptr->dim() == 3, "batch2 must be a 3D tensor");
  int64_t ntensors = self_ptr->size(0),
      ntensors2 = mat2_ptr->size(0);
  TORCH_CHECK(ntensors == ntensors2,
      "Expected size for the 1st dimension of batch2 tensor to be: ", ntensors,
      " but got: ", ntensors2, ".");
  const Tensor& self_buffer = self_ptr->get_buffer(),
      & mat2_buffer = mat2_ptr->get_buffer();
  std::vector<IntArrayRef> self_sizes = NestedTensor_get_sizes(self_ptr),
      mat2_sizes = NestedTensor_get_sizes(mat2_ptr),
      self_strides = NestedTensor_get_strides(self_ptr),
      mat2_strides = NestedTensor_get_strides(mat2_ptr);
  const std::vector<int64_t>& self_offsets = self_ptr->get_offsets(),
      & mat2_offsets = mat2_ptr->get_offsets();
  // create a contiguous output
  int64_t out_numel = 0;
  const Tensor& self_sizemat = self_ptr->get_nested_size_tensor();
  Tensor out_sizemat = self_sizemat.new_empty(self_sizemat.sizes());
  int64_t* out_sizemat_ptr = out_sizemat.data_ptr<int64_t>();
  for (int64_t i = 0; i < ntensors; i++) {
    const IntArrayRef& self_shape = self_sizes[i],
        & mat2_shape = mat2_sizes[i];
    const int64_t& self_size0 = self_shape[0], & self_size1 = self_shape[1],
        & mat2_size0 = mat2_shape[0], & mat2_size1 = mat2_shape[1];
    TORCH_CHECK(self_size1 == mat2_size0,
        i, "-th nested matrices in batch cannot be multiplied (",
        self_size0, "x", self_size1, " and ",
        mat2_size0, "x", mat2_size1, ")");
    out_sizemat_ptr[0] = self_size0;
    out_sizemat_ptr[1] = mat2_size1;
    out_sizemat_ptr += 2;
    out_numel += self_size0 * mat2_size1;
  }
  Tensor out_buffer = self_buffer.new_empty(out_numel);
  Tensor output = wrap_buffer(out_buffer, out_sizemat);
  // call tensor mm
  // TODO: `padding nested tensor -> bmm -> remove padding` may be more efficient
  //       until we have specialized nested tensor bmm kernel
  //       useful resource: `aten/src/ATen/native/cpu/LinearAlgebra.cpp/bmm_out_or_baddbmm_`
  //                        `aten/src/ATen/native/cuda/Blas.cpp/baddbmm_out_cuda_impl`
  std::vector<Tensor> output_unbind = output.unbind();
  for (int64_t i = 0; i < ntensors; i++) {
    at::mm_out(output_unbind[i],
               self_buffer.as_strided(self_sizes[i], self_strides[i], self_offsets[i]),
               mat2_buffer.as_strided(mat2_sizes[i], mat2_strides[i], mat2_offsets[i]));
  }
  return output;
}

// utilities support _NestedTensor_GeneralizedBMM
namespace {
inline std::tuple<std::vector<int64_t>, Tensor>
_NestedTensor_GeneralizedBMM_BatchSizes_OutputMemory(
    const std::vector<IntArrayRef>& self_sizes,
    const std::vector<IntArrayRef>& mat2_sizes,
    const c10::TensorOptions& buffer_op,
    const c10::TensorOptions& sizemat_op) {
  int64_t ntensors = self_sizes.size(),
      ndims = self_sizes[0].size();
  std::vector<int64_t> batch_sizes(ntensors, 1);
  Tensor sizemat = at::empty({ntensors, ndims}, sizemat_op);
  int64_t* sizemat_ptr = sizemat.data_ptr<int64_t>();
  int64_t numel = 0;
  for (int64_t i = 0; i < ntensors; i++) {
    const IntArrayRef& self_size = self_sizes[i],
        & mat2_size = mat2_sizes[i];
    int64_t& batch_size = batch_sizes[i];
    // batch dimensions
    for (int64_t j = 0; j < ndims - 2; j++) {
      const int64_t& self_sizej = self_size[j],
          & mat2_sizej = mat2_size[j];
      TORCH_CHECK(
          self_sizej == mat2_sizej,
          "matmul: For nested tensors, no broadcasting is currently performed: ",
          i, "-th nested matrices in batch at dimension ", j + 1,
          " have mismatching sizes ", self_sizej, " and ", mat2_sizej);
      sizemat_ptr[j] = self_sizej;
      batch_size *= sizemat_ptr[j];
    }
    // matrix multiplication dimensions
    const int64_t& self_size0 = self_size[ndims - 2], & self_size1 = self_size[ndims - 1],
        & mat2_size0 = mat2_size[ndims - 2], & mat2_size1 = mat2_size[ndims - 1];
    TORCH_CHECK(
        self_size1 == mat2_size0,
        "matmul: ",
        i, "-th nested matrices in batch cannot be multiplied (",
        self_size0, "x", self_size1, " and ",
        mat2_size0, "x", mat2_size1, ")");
    sizemat_ptr[ndims - 2] = self_size0;
    sizemat_ptr[ndims - 1] = mat2_size1;
    sizemat_ptr += ndims;
    numel += batch_size * self_size0 * mat2_size1;
  }
  Tensor buffer = at::empty(numel, buffer_op);
  Tensor output = wrap_buffer(buffer, sizemat);
  return std::make_tuple(batch_sizes, output);
}
}

// This is a generalized batched matmul dedicated to nested tensors,
// where `self` and `mat2` have same number (>= 3) of dimensions.
// The last 2 dimensions will be considered as matrix dimensions,
// so they should be matrix-multiplicable.
// The leading dimensions are considered as batch dimensions,
// and since nested tensor does not support broadcasting for now,
// for each batch dimension `self` and `mat2` must have same size.
Tensor _NestedTensor_GeneralizedBMM(const Tensor& self, const Tensor& mat2) {
  auto self_ptr = get_nested_tensor_impl(self),
      mat2_ptr = get_nested_tensor_impl(mat2);
  int64_t self_dim = self_ptr->dim(),
      mat2_dim = mat2_ptr->dim();
  TORCH_CHECK(
      self_dim >= 3,
      "matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: ",
      self_dim);
  TORCH_CHECK(
      mat2_dim >= 3,
      "matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 2nd input has rank: ",
      mat2_dim);
  TORCH_CHECK(self_dim == mat2_dim, "matmul: both inputs must have same rank");
  int64_t ntensors = self_ptr->size(0),
      ntensors2 = mat2_ptr->size(0);
  TORCH_CHECK(ntensors == ntensors2,
      "matmul: Expected size for the 1st dimension of 2nd input tensor to be: ", ntensors,
      " but got: ", ntensors2, ".");
  const Tensor& self_buffer = self_ptr->get_buffer(),
      & mat2_buffer = mat2_ptr->get_buffer();
  std::vector<IntArrayRef> self_sizes = NestedTensor_get_sizes(self_ptr),
      mat2_sizes = NestedTensor_get_sizes(mat2_ptr),
      self_strides = NestedTensor_get_strides(self_ptr),
      mat2_strides = NestedTensor_get_strides(mat2_ptr);
  const std::vector<int64_t>& self_offsets = self_ptr->get_offsets(),
      & mat2_offsets = mat2_ptr->get_offsets();
  // create a contiguous output
  std::vector<int64_t> batch_sizes;
  Tensor output;
  std::tie(batch_sizes, output) = _NestedTensor_GeneralizedBMM_BatchSizes_OutputMemory(
      self_sizes, mat2_sizes, self_buffer.options(), self_ptr->get_nested_size_tensor().options());
  // call tensor matmul
  // TODO: `padding nested tensor -> bmm -> remove padding` may be more efficient
  //       until we have specialized nested tensor bmm kernel
  //       useful resource: `aten/src/ATen/native/cpu/LinearAlgebra.cpp/bmm_out_or_baddbmm_`
  //                        `aten/src/ATen/native/cuda/Blas.cpp/baddbmm_out_cuda_impl`
  std::vector<Tensor> output_unbind = output.unbind();
  for (int64_t i = 0; i < ntensors; i++) {
    const IntArrayRef& self_size = self_sizes[i],
        & mat2_size = mat2_sizes[i];
    const int64_t& batch_size = batch_sizes[i];
    if (batch_size == 1) {
      at::mm_out(
          output_unbind[i],
          self_buffer.as_strided(self_size, self_strides[i], self_offsets[i]),
          mat2_buffer.as_strided(mat2_size, mat2_strides[i], mat2_offsets[i])
          );
    }
    else {
      at::bmm_out(
          output_unbind[i],
          self_buffer.as_strided(self_size, self_strides[i], self_offsets[i])
              .reshape({batch_size, self_size[self_dim - 1 - 2], self_size[self_dim - 1 - 1]}),
          mat2_buffer.as_strided(mat2_size, mat2_strides[i], mat2_offsets[i])
              .reshape({batch_size, mat2_size[self_dim - 1 - 2], mat2_size[self_dim - 1 - 1]})
          );
    }
  }
  return output;
}

Tensor transpose_nested(const Tensor& self, int64_t dim0, int64_t dim1) {
  auto self_ptr = get_nested_tensor_impl(self);
  // check input dimensions
  int64_t ndims = self_ptr->dim();
  int64_t positive_dim0 = at::maybe_wrap_dim(dim0, ndims),
      positive_dim1 = at::maybe_wrap_dim(dim1, ndims);
  if (positive_dim0 == positive_dim1) {
    return self;
  }
  TORCH_CHECK(positive_dim0 > 0 && positive_dim1 > 0, "Nested tensor dimension 0 cannot be transposed");
  // -- to exclude the implicit batch dimension
  ndims--;
  positive_dim0--;
  positive_dim1--;
  // transpose = switch `dim0` and `dim1` columns of `sizemat` and `stridemat`
  const Tensor& sizemat = self_ptr->get_nested_size_tensor(),
      & stridemat = self_ptr->get_nested_stride_tensor();
  Tensor column_indices = sizemat.new_empty(ndims);
  int64_t* column_indices_ptr = column_indices.data_ptr<int64_t>();
  std::iota(column_indices_ptr, column_indices_ptr + ndims, 0);
  column_indices_ptr[positive_dim0] = positive_dim1;
  column_indices_ptr[positive_dim1] = positive_dim0;
  // create transposed `sizemat` and `stridemat`
  Tensor sizemat_transposed = at::index_select(sizemat, 1, column_indices),
      stridemat_transposed = at::index_select(stridemat, 1, column_indices);
  return wrap_buffer(self_ptr->get_buffer(), sizemat_transposed, stridemat_transposed, self_ptr->get_offsets());
}

// utilities supporting `reshape`
namespace {
// Args:
//     sizes: the sizes of original nested tensor
//     strides: the strides of original nested tensor
//     proposed_shape: user proposed new shape
//     op: the options for new size and stride matrices
// Returns:
//     whether reshape as view is possible (i.e. old buffer can be reused)
//     size matrix after reshape
//     stride matrix after reshape (not fully populated if reshape as view is impossible)
inline std::tuple<bool, Tensor, Tensor> NestedTensor_reshape_size_stride(
    const std::vector<IntArrayRef>& sizes,
    const std::vector<IntArrayRef>& strides,
    const IntArrayRef& proposed_shape,
    const c10::TensorOptions& op) {
  int64_t ntensors = sizes.size(),
      ndims_underlying = sizes[0].size(),
      ndims_underlying_reshaped = proposed_shape.size() - 1;
  bool reshape_as_view = true;
  Tensor sizemat_reshaped = at::empty({ntensors, ndims_underlying_reshaped}, op),
      stridemat_reshaped = at::empty({ntensors, ndims_underlying_reshaped}, op);
  int64_t* sizemat_reshaped_ptr = sizemat_reshaped.data_ptr<int64_t>(),
      * stridemat_reshaped_ptr = stridemat_reshaped.data_ptr<int64_t>();
  for (int64_t itensor = 0; itensor < ntensors; itensor++) {
    const IntArrayRef& size = sizes[itensor],
        & stride = strides[itensor];
    // compute reshaped size
    std::vector<int64_t> size_reshaped_vector(proposed_shape.begin() + 1, proposed_shape.end());
    // some negative sizes remain to be infered
    if (ndims_underlying < ndims_underlying_reshaped) {
      // replace negative sizes for old dimensions with old sizes
      int64_t numel = 1, numel_reshaped = 1;
      for (int64_t idim = 0; idim < ndims_underlying; idim++) {
        int64_t& size_reshaped = size_reshaped_vector[idim];
        TORCH_CHECK(size_reshaped >= -1, "invalid shape dimension ", size_reshaped);
        if (size_reshaped == -1) {
          size_reshaped = size[idim];
        }
        numel *= size[idim];
        numel_reshaped *= size_reshaped;
      }
      // infer negative size for new dimension
      int64_t infer_index = -1;
      for (int64_t idim = ndims_underlying; idim < ndims_underlying_reshaped; idim++) {
        const int64_t& size_reshaped = size_reshaped_vector[idim];
        if (size_reshaped >= 0) {
          numel_reshaped *= size_reshaped;
        }
        else if (size_reshaped == -1) {
          if (infer_index > -1) {
            throw std::runtime_error("only one dimension can be inferred");
          }
          else {
            infer_index = idim;
          }
        }
        else {
          AT_ERROR("invalid shape dimension ", size_reshaped);
        }
      }
      // See Note [inference and inheritance semantics]
      TORCH_CHECK(infer_index == -1, "nested tensor does not infer shape");
    }
    // all negative sizes can be replaced
    else {
      for (int64_t idim = 0; idim < ndims_underlying_reshaped; idim++) {
        int64_t& size_reshaped = size_reshaped_vector[idim];
        TORCH_CHECK(size_reshaped >= -1, "invalid shape dimension ", size_reshaped);
        if (size_reshaped == -1) {
          size_reshaped = size[idim];
        }
      }
    }
    IntArrayRef size_reshaped(size_reshaped_vector);
    // compute reshaped stride
    auto opt_stride_reshaped = at::detail::computeStride(size, stride, size_reshaped);
    // reshape as view is possible
    if (opt_stride_reshaped.has_value()) {
      const IntArrayRef& stride_reshaped = *opt_stride_reshaped;
      // fill reshaped size and stride into sizemat and stridemat
      for (int64_t idim = 0; idim < ndims_underlying_reshaped; idim++) {
        sizemat_reshaped_ptr[idim] = size_reshaped[idim];
        stridemat_reshaped_ptr[idim] = stride_reshaped[idim];
      }
      sizemat_reshaped_ptr += ndims_underlying_reshaped;
      stridemat_reshaped_ptr += ndims_underlying_reshaped;
    }
    // reshape as view is impossible
    else {
      reshape_as_view = false;
      // fill reshaped size into sizemat
      for (int64_t idim = 0; idim < ndims_underlying_reshaped; idim++) {
        sizemat_reshaped_ptr[idim] = size_reshaped[idim];
      }
      sizemat_reshaped_ptr += ndims_underlying_reshaped;
    }
  }
  return std::make_tuple(reshape_as_view, sizemat_reshaped, stridemat_reshaped);
}

// Args:
//     nt_reshaped: the reshaped nested tensor to receive copies
//     buffer: the original nested tensor buffer
//     sizes: the original nested tensor sizes (may have gone through collapsing or splitting)
//     strides: the original nested tensor strides (may have gone through collapsing or splitting)
//     offsets: the original nested tensor offsets (may have gone through collapsing or splitting)
inline void NestedTensor_reshape_copy(
    Tensor& nt_reshaped,
    const Tensor& buffer,
    const std::vector<IntArrayRef>& sizes,
    const std::vector<IntArrayRef>& strides,
    const std::vector<int64_t>& offsets) {
    auto nt_reshaped_ptr = get_nested_tensor_impl(nt_reshaped);
    const Tensor& buffer_reshaped = nt_reshaped_ptr->get_buffer();
    std::vector<IntArrayRef> sizes_reshaped = NestedTensor_get_sizes(nt_reshaped_ptr),
        strides_reshaped = NestedTensor_get_strides(nt_reshaped_ptr);
    const std::vector<int64_t>& offsets_reshaped = nt_reshaped_ptr->get_offsets();
    for (int64_t i = 0; i < nt_reshaped_ptr->size(0); i++) {
      buffer_reshaped.as_strided(sizes_reshaped[i], strides_reshaped[i], offsets_reshaped[i]).copy_(
          // TODO: can we avoid allocating new memory for `buffer...reshape`
          //       I did not find anything like reshape_out
          buffer.as_strided(sizes[i], strides[i], offsets[i]).reshape(sizes_reshaped[i]));
    }
}
} // namespace

// Special rules for reshape(nested tensor):
// 1. Only 1 regular dimension can be collapsed with
//    or splitted from the implicit batch dimension
// 2. Instead of infering size, -1 means "inherit the old size", so:
//    * negative size is legal for a ragged dimension
//    * multiple sizes can be -1
Tensor reshape_nested(const Tensor& self, IntArrayRef proposed_shape) {
  TORCH_CHECK(
      proposed_shape.size() > 0,
      "shape '[]' is invalid for a nested tensor");
  auto self_ptr = get_nested_tensor_impl(self);
  // basic information before reshaping
  int64_t ntensors = self_ptr->size(0);
  TORCH_CHECK(
      ntensors > 0,
      "empty nested tensor cannot be reshaped");
  // basic information after reshaping
  int64_t ntensors_reshaped;
  if (proposed_shape[0] >= 0) {
    ntensors_reshaped = proposed_shape[0];
  }
  else if (proposed_shape[0] == -1) {
    ntensors_reshaped = ntensors;
  }
  else {
    AT_ERROR("invalid shape dimension ", proposed_shape[0]);
  }
  TORCH_CHECK(
      ntensors == ntensors_reshaped,
      "for now reshape cannot change the implicit batch dimension");
  std::vector<IntArrayRef> sizes = NestedTensor_get_sizes(self_ptr),
      strides = NestedTensor_get_strides(self_ptr);
  const std::vector<int64_t>& offsets = self_ptr->get_offsets();
  // reshaping underlying tensor dimensions does not change offset
  // determine reshaped size and stride
  const Tensor& buffer = self_ptr->get_buffer(),
      & sizemat = self_ptr->get_nested_size_tensor();
  bool reshape_as_view;
  Tensor sizemat_reshaped, stridemat_reshaped;
  std::tie(reshape_as_view, sizemat_reshaped, stridemat_reshaped) = NestedTensor_reshape_size_stride(
      sizes, strides, proposed_shape, sizemat.options());
  if (reshape_as_view) {
    return wrap_buffer(buffer, sizemat_reshaped, stridemat_reshaped, offsets);
  }
  Tensor buffer_reshaped = buffer.new_empty(buffer.sizes());
  Tensor output = wrap_buffer(buffer_reshaped, sizemat_reshaped);
  NestedTensor_reshape_copy(output,
      buffer, sizes, strides, offsets);
  return output;
}

} // namespace native
} // namespace at
