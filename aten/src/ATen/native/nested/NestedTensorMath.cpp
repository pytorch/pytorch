#include <ATen/native/nested/NestedTensorMath.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/TensorShape.h>
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
  for (const int64_t i: c10::irange(ntensors)){
    result_tensors[i] = buffer.as_strided(sizes[i], strides[i], offsets[i]);
  }
  return result_tensors;
}

Tensor& NestedTensor_relu_(Tensor& self) {
  auto buffer = get_nested_tensor_impl(self)->get_buffer();
  at::relu_(buffer);
  return self;
}

Tensor NestedTensor_relu(const Tensor& self) {
  return map_nt(self, at::relu);
}

Tensor& NestedTensor_gelu_(Tensor& self, c10::string_view approximate) {
  auto buffer = get_nested_tensor_impl(self)->get_buffer();
  at::gelu_(buffer, approximate);
  return self;
}

Tensor NestedTensor_gelu(const Tensor& self, c10::string_view approximate) {
  return map_nt(
      self,
      [approximate](const Tensor& buffer) {
        return at::gelu(buffer, approximate);
      });
}

Tensor NestedTensor_nested_tensor_from_mask(const Tensor& t, const Tensor& mask, bool mask_check) {
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

    if (mask_check)
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
  }
  return impl::wrap_tensor_node(
      impl::TensorNode(list),
      dtype,
      layout,
      device,
      pin_memory);
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

// Only usable on the C++ side; scalars are converted to tensors coming from Python.
Tensor NestedTensor_mul_Scalar(const Tensor& self, const Scalar& other) {
  return NestedTensor_mul_Tensor(self, wrapped_scalar_tensor(other));
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

// Only usable on the C++ side; scalars are converted to tensors coming from Python.
Tensor& NestedTensor_mul__Scalar(Tensor& self, const Scalar& other) {
  return NestedTensor_mul__Tensor(self, wrapped_scalar_tensor(other));
}

// Very rudimentary sum_dim for prototyping with torch_scatter.segment_reduce.
Tensor NestedTensor_sum_dim_CPU(
    const Tensor& self,
    OptionalIntArrayRef opt_dims,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  // Only allow reductions across the last dim
  auto dims = opt_dims.value_or(IntArrayRef{});
  TORCH_CHECK(
      dims.size() == 1,
      "NestedTensor only allows reduction of a single dimension for now."
  );
  auto dim = maybe_wrap_dim(dims[0], self.dim());
  TORCH_CHECK(
      dim == self.dim() - 1,
      "NestedTensor can only be reduced across the last dimension for now ",
      "got dimension ",
      dim,
      " instead.");
  // Always keep reduced dim for now
  // This is to avoid the case where the nested tensors are 1D and keepdim=False
  // making the nested tensors -> elements (e.g. sum(nt([1, 2 ,3], [4, 5]), -1) -> nt(6, 9))
  TORCH_CHECK(keepdim, "NestedTensor always requires keepdim=True for now.");
  // acc_dtype is not supported for now
  TORCH_CHECK(!dtype, "NestedTensor does not support dtype argument for now.");

  auto nt_input = get_nested_tensor_impl(self);
  TORCH_CHECK(
      nested_tensor_impl_is_contiguous(nt_input),
      "NestedTensor does not support reductions when the input is noncontiguous for now.");
  int64_t ntensors = nt_input->size(0);
  if (ntensors == 0) {
    return self;
  }
  const Tensor& buffer = nt_input->get_buffer();

  auto sizemat = nt_input->get_nested_size_tensor();
  // create output size tensor for keepdim=True
  auto output_sizemat = sizemat.clone();
  output_sizemat.select(1, -1).fill_(1);

  auto num_segments = at::prod(output_sizemat, -1);
  auto segment_lengths = sizemat.select(1, -1);
  const int64_t new_numel = at::sum(num_segments).item<int64_t>();
  auto output_buffer = buffer.new_empty(IntArrayRef(new_numel));

  // This logic assumes for now that
  // (1) all the nested tensors are contiguous
  // (2) the nested tensors are stored contiguously in the buffer
  AT_DISPATCH_ALL_TYPES_AND2(
    ScalarType::Half, ScalarType::BFloat16, buffer.scalar_type(), "nested_sum_dim_cpu", [&]() {
    auto* output_data = output_buffer.data_ptr<scalar_t>();
    const auto* input_data = buffer.data_ptr<scalar_t>();
    int64_t out_idx = 0, in_idx = 0;
    for (const auto i : c10::irange(ntensors)) {
      int64_t segments = num_segments[i].item<int64_t>();
      int64_t segment_length = segment_lengths[i].item<int64_t>();
      for (auto j = 0; j < segments; j++) {
        scalar_t res = 0;
        for (auto k = 0; k < segment_length; k++) {
          res += input_data[in_idx];
          in_idx += 1;
        }
        output_data[out_idx] = res;
        out_idx += 1;
      }
    }
  });

  return wrap_buffer(output_buffer, output_sizemat);
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
  auto memory_format = optional_memory_format.value_or(c10::MemoryFormat::Preserve);
  auto self_ptr = get_nested_tensor_impl(self);
  if (memory_format == c10::MemoryFormat::Preserve ||
  (memory_format == c10::MemoryFormat::Contiguous && nested_tensor_impl_is_contiguous(self_ptr))) {
    const Tensor& buffer = self_ptr->get_buffer(),
        sizemat = self_ptr->get_nested_size_tensor(),
        stridemat = self_ptr->get_nested_stride_tensor();
    const std::vector<int64_t>& offsets = self_ptr->get_offsets();
    // TODO: The size and the stride do not necessarily need to be cloned,
    //       but it is more conservative.
    //       This is something we could revisit once we land a more
    //       efficient implementation of nested_size_tensor_ and nested_stride_tensor.
    return wrap_buffer(buffer.clone(), sizemat.clone(), stridemat.clone(), offsets);
  }
  // actually, memory format is contiguous and self is noncontiguous
  else if (memory_format == c10::MemoryFormat::Contiguous) {
    const Tensor& self_buffer = self_ptr->get_buffer(),
        sizemat = self_ptr->get_nested_size_tensor();
    // TODO: use self.empty_like() rather than self.new_empty(self.sizes())
    //       once nested-tensor empty_like comes out
    Tensor output_buffer = self_buffer.new_empty(self_buffer.sizes());
    Tensor output = wrap_buffer(output_buffer, sizemat);
    std::vector<Tensor> self_unbind = self.unbind(),
        output_unbind = output.unbind();
    for (int64_t i = 0; i < self_ptr->size(0); i++) {
      output_unbind[i].copy_(self_unbind[i]);
    }
    return output;
  }
  else {
    AT_ERROR("nested tensor clone supports Preserve and Contiguous memory foramts");
  }
}

at::Tensor NestedTensor_get_nested_size_tensor(const at::Tensor& self){
  return get_nested_size_tensor(self);
}

std::tuple<Tensor,Tensor> native_dropout_nested(const Tensor& input, double p, c10::optional<bool> train) {
  auto input_ptr = get_nested_tensor_impl(input);
  const Tensor& input_buffer = input_ptr->get_buffer(),
      & sizemat = input_ptr->get_nested_size_tensor(),
      & stridemat = input_ptr->get_nested_stride_tensor();
  const std::vector<int64_t>& offsets = input_ptr->get_offsets();
  Tensor output_buffer, mask_buffer;
  if (input_buffer.numel() == 0) {
    output_buffer = input_buffer.clone();
    mask_buffer = input_buffer.clone();
  }
  else {
    std::tie(output_buffer, mask_buffer) = at::native_dropout(input_buffer, p, train);
  }
  // regular tensor dropout reuses input size and stride
  // i.e. if input is not contiguous, then output is also discontiguous
  Tensor output = wrap_buffer(output_buffer, sizemat.clone(), stridemat.clone(), offsets),
      mask = wrap_buffer(mask_buffer, sizemat.clone(), stridemat.clone(), offsets);
  return std::make_tuple(output, mask);
}

Tensor softmax_nested(
    const Tensor& input,
    const int64_t dim,
    const bool half_to_float) {
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
  if (self.is_nested() && !mat2.is_nested()) {
    AT_ERROR("Expected both to be nested, but got a nested self and non-nested other");
  }
  else if (!self.is_nested() && mat2.is_nested()) {
    AT_ERROR("Expected both to be nested, but got a non-nested self and nested other");
  }
  // dispatcher should have guaranteed that at least one is nested
  auto self_ptr = get_nested_tensor_impl(self);
  auto mat2_ptr = get_nested_tensor_impl(mat2);
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

// utilities support `matmul_nested`
namespace {
// Args:
//     self_sizes: the sizes of `self` in `matmul_nested`
//     mat2_sizes: the sizes of `mat2` in `matmul_nested`
//     buffer_op: the options for new buffer
//     sizemat_op: the options for new size matrix
// Returns:
//     the batch size of each input underlying tensor, i.e. the product of batch-dimension sizes
//     the empty output nested tensor
inline std::tuple<std::vector<int64_t>, Tensor>
matmul_nested_helper(
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

// Note [nested tensor matmul]
// This is really a generalized batched matmul dedicated to nested tensors,
// where `self` and `mat2` have same number (>= 3) of dimensions.
// The last 2 dimensions will be considered as matrix dimensions,
// so they should be matrix-multiplicable.
// The leading dimensions are considered as batch dimensions,
// and since nested tensor does not support broadcasting for now,
// for each batch dimension `self` and `mat2` must have same size.
// TODO: Should make full matmul semantics support some day
Tensor matmul_nested(const Tensor& self, const Tensor& mat2) {
  if (self.is_nested() && !mat2.is_nested()) {
    AT_ERROR("Expected both to be nested, but got a nested self and non-nested other");
  }
  else if (!self.is_nested() && mat2.is_nested()) {
    AT_ERROR("Expected both to be nested, but got a non-nested self and nested other");
  }
  // dispatcher should have guaranteed that at least one is nested
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
  std::tie(batch_sizes, output) = matmul_nested_helper(
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

Tensor& matmul_out_nested(const Tensor& tensor1, const Tensor& tensor2, Tensor& result) {
  // TODO: this is a very quick and dirty implementation
  //       should improve it to avoid the intermediate memory usage
  Tensor function_result = at::matmul(tensor1, tensor2);
  auto function_result_ptr = get_nested_tensor_impl(function_result);
  // TODO: this is to reproduce function_result_ptr->opt_sizes_
  //       if an accessor is provided in the future, can replace this
  std::vector<int64_t> sizes;
  for (int64_t i = 0; i < function_result_ptr->dim(); i++) {
    c10::optional<int64_t> opt_size = function_result_ptr->opt_size(i);
    if (opt_size.has_value()) {
      sizes.push_back(*opt_size);
    }
    else {
      sizes.push_back(-1);
    }
  }
  result.reshape(sizes);
  result.copy_(function_result);
  return result;
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
  return create_nested_view_tensor(
      self, sizemat_transposed, stridemat_transposed, std::vector<int64_t>(self_ptr->get_offsets()));
}

// utilities supporting `_reshape_nested`
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
      for (int64_t idim = 0; idim < ndims_underlying; idim++) {
        int64_t& size_reshaped = size_reshaped_vector[idim];
        TORCH_CHECK(size_reshaped >= -1, "invalid shape dimension ", size_reshaped);
        if (size_reshaped == -1) {
          size_reshaped = size[idim];
        }
      }
      // infer negative size for new dimension
      int64_t infer_index = -1;
      for (int64_t idim = ndims_underlying; idim < ndims_underlying_reshaped; idim++) {
        const int64_t& size_reshaped = size_reshaped_vector[idim];
        if (size_reshaped == -1) {
          if (infer_index > -1) {
            throw std::runtime_error("only one dimension can be inferred");
          }
          else {
            infer_index = idim;
          }
        }
        else if (size_reshaped < 0) {
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
Tensor _reshape_nested(const Tensor& self, IntArrayRef proposed_shape) {
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

Tensor slice_nested(
    const Tensor& self,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step) {
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "slice() cannot be applied to a 0-dim tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);

  // will error if the dimension is jagged
  auto dim_size = self.size(dim);

  TORCH_CHECK(step > 0, "slice step must be positive");

  auto* nt_impl = get_nested_tensor_impl(self);
  const int64_t ntensors = nt_impl->size(0);
  const auto& sizes = nt_impl->get_nested_size_tensor();
  const auto& strides = nt_impl->get_nested_stride_tensor();
  const std::vector<int64_t>& offsets = nt_impl->get_offsets();

  // special case for empty NT
  if (ntensors == 0) {
    return create_nested_view_tensor(self, sizes.clone(), strides.clone(), std::vector<int64_t>(offsets));
  }

  int64_t start_val, end_val;
  std::tie(start_val, end_val) = get_slice_range(start, end, dim_size);
  auto len = end_val - start_val;

  // special case for slicing the tensor components themselves when dim=0
  auto new_sizes = (dim > 0) ? sizes.clone() : sizes.slice(0, start_val, end_val).clone();
  auto new_strides = (dim > 0) ? strides.clone() : strides.slice(0, start_val, end_val).clone();
  auto new_offsets = (dim > 0) ? std::vector<int64_t>(offsets) :
      std::vector<int64_t>(offsets.begin() + start_val, offsets.begin() + end_val);

  if (dim > 0) {
    // account for the implicit batch dim
    --dim;
    for (int64_t i : c10::irange(ntensors)) {
      int64_t *size_ptr = new_sizes[i].data_ptr<int64_t>();
      int64_t *stride_ptr = new_strides[i].data_ptr<int64_t>();

      new_offsets[i] = offsets[i] + start_val * stride_ptr[dim];
      size_ptr[dim] = (len + step - 1) / step; // round-up
      stride_ptr[dim] *= step;
    }
  }

  return create_nested_view_tensor(self, new_sizes, new_strides, std::vector<int64_t>(new_offsets));
}

} // namespace native
} // namespace at
