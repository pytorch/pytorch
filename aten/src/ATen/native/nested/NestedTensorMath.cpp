#include <ATen/native/nested/NestedTensorMath.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#include <tuple>


namespace at {
namespace native {

namespace {
template <typename Func>
Tensor map_nt(const Tensor& nt, Func f) {
  auto* nt_impl = get_nested_tensor_impl(nt);
  const auto& sizes = nt_impl->get_nested_size_tensor();
  return at::detail::make_tensor<NestedTensorImpl>(f(nt_impl->get_buffer()), sizes);
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
  // This returns a differentiable view of self as a regular tensor
  auto buffer = self.values();
  std::vector<IntArrayRef> sizes = NestedTensor_get_sizes(self_ptr),
      strides = NestedTensor_get_strides(self_ptr);
  const std::vector<int64_t>& offsets = self_ptr->get_storage_offsets();
  for (const int64_t i: c10::irange(ntensors)){
    result_tensors[i] = buffer.as_strided(sizes[i], strides[i], offsets[i]);
  }
  return result_tensors;
}

Tensor& NestedTensor_relu_(Tensor& self) {
  auto self_ptr = get_nested_tensor_impl(self);
  check_numel_equals_buffer_size(self_ptr);
  auto buffer = self_ptr->get_buffer();
  at::relu_(buffer);
  return self;
}

Tensor NestedTensor_relu(const Tensor& self) {
  return map_nt(self, at::relu);
}

Tensor& NestedTensor_gelu_(Tensor& self, c10::string_view approximate) {
  auto self_ptr = get_nested_tensor_impl(self);
  check_numel_equals_buffer_size(self_ptr);
  auto buffer = self_ptr->get_buffer();
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

Tensor& NestedTensor_tanh_(Tensor& self) {
  auto self_ptr = get_nested_tensor_impl(self);
  check_numel_equals_buffer_size(self_ptr);
  auto buffer = self_ptr->get_buffer();
  at::tanh_(buffer);
  return self;
}

Tensor NestedTensor_tanh(const Tensor& self) {
  return map_nt(self, at::tanh);
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

Tensor _nested_tensor_from_tensor_list(
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

C10_ALWAYS_INLINE std::pair<int64_t, int64_t> _check_nested_layer_norm_inputs(
    const NestedTensorImpl& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */) {

  const size_t normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight.defined() || weight.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !bias.defined() || bias.sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias.sizes(),
      " and normalized_shape = ",
      normalized_shape);

  // Check that the normalized_shape has the exact same sizes as the last dimensions from the NestedTensor input
  // Also, compute M and N considering the idiosyncracies of NestedTensors
  int64_t N = 1;
  for (const auto i: c10::irange(normalized_ndim)) {
    TORCH_CHECK(
      input.opt_size(-normalized_ndim + i) != c10::nullopt,
      "normalized_shape extends into irregular dimensions for the nested tensor"
    );
    TORCH_CHECK(
      normalized_shape[i] == *input.opt_size(-normalized_ndim + i),
      "The shape at dimension ",
      i,
      "of normalized_shape doesn't match the input"
    );
    N *= normalized_shape[i];
  }

  const int64_t M = input.numel() / N;

  return std::make_pair(M, N);
}

std::tuple<Tensor, Tensor, Tensor> nested_layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape,
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
  auto M_N = _check_nested_layer_norm_inputs(*nt_input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
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
  Tensor mean = at::empty({M}, options);
  Tensor rstd = at::empty({M}, options);
  LayerNormKernel(
      input_buffer.is_cuda() ? kCUDA : kCPU,
      input_buffer,
      *weight_contig,
      *bias_contig,
      M,
      N,
      eps,
      &output_buffer,
      &mean,
      &rstd);
  return std::make_tuple(
    wrap_buffer(output_buffer, nt_input->get_nested_size_tensor()),
    mean,
    rstd
  );
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
  const at::Tensor& buffer = self_ptr->get_unsafe_storage_as_tensor();
  std::vector<IntArrayRef> sizes = NestedTensor_get_sizes(self_ptr),
      strides = NestedTensor_get_strides(self_ptr);
  const std::vector<int64_t>& offsets = self_ptr->get_storage_offsets();
  return buffer.as_strided(sizes[positive_index], strides[positive_index], offsets[positive_index]);
}

Tensor clone_nested(
    const Tensor& self,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto memory_format = optional_memory_format.value_or(c10::MemoryFormat::Preserve);
  auto self_ptr = get_nested_tensor_impl(self);
  if (memory_format == c10::MemoryFormat::Preserve ||
  (memory_format == c10::MemoryFormat::Contiguous && self.is_contiguous())) {
    const Tensor& buffer = self_ptr->get_unsafe_storage_as_tensor(),
        sizemat = self_ptr->get_nested_size_tensor(),
        stridemat = self_ptr->get_nested_stride_tensor();
    const std::vector<int64_t>& offsets = self_ptr->get_storage_offsets();
    // TODO: The size and the stride do not necessarily need to be cloned,
    //       but it is more conservative.
    //       This is something we could revisit once we land a more
    //       efficient implementation of nested_size_tensor_ and nested_stride_tensor.
    return wrap_buffer(buffer.clone(), sizemat.clone(), stridemat.clone(), std::vector<int64_t>(offsets));
  }
  // actually, memory format is contiguous and self is noncontiguous
  else if (memory_format == c10::MemoryFormat::Contiguous) {
    const Tensor& self_buffer = self_ptr->get_unsafe_storage_as_tensor(),
        sizemat = self_ptr->get_nested_size_tensor();
    Tensor output_buffer = at::empty(self.numel(), self_buffer.options());
    Tensor output = wrap_buffer(output_buffer, sizemat);
    std::vector<Tensor> self_unbind = self.unbind(),
        output_unbind = output.unbind();
    for (const int64_t i: c10::irange(self_ptr->size(0))) {
      output_unbind[i].copy_(self_unbind[i]);
    }
    return output;
  } else {
    TORCH_CHECK(
        false,
        "Nested tensor clone supports Preserve and Contiguous memory formats, called clone with memory format: ",
        memory_format);
  }
}

std::tuple<Tensor,Tensor> native_dropout_nested(const Tensor& input, double p, c10::optional<bool> train) {
  auto input_ptr = get_nested_tensor_impl(input);
  const Tensor& input_buffer = input_ptr-> get_unsafe_storage_as_tensor(),
      & sizemat = input_ptr->get_nested_size_tensor(),
      & stridemat = input_ptr->get_nested_stride_tensor();
  const std::vector<int64_t>& offsets = input_ptr->get_storage_offsets();
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
  Tensor output = wrap_buffer(output_buffer, sizemat.clone(), stridemat.clone(), std::vector<int64_t>(offsets)),
      mask = wrap_buffer(mask_buffer, sizemat.clone(), stridemat.clone(), std::vector<int64_t>(offsets));
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
  // TODO We would ideally use a empty_like here, but that is not supported
  // for nested tensors yet. Since we are only using the buffer for the options
  // and size it is okay to use unsafe_storage_as_tensor here.
  const Tensor& buffer = input_ptr->get_unsafe_storage_as_tensor(),
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
  const Tensor& self_buffer = self_ptr->get_unsafe_storage_as_tensor(),
      & mat2_buffer = mat2_ptr->get_unsafe_storage_as_tensor();
  std::vector<IntArrayRef> self_sizes = NestedTensor_get_sizes(self_ptr),
      mat2_sizes = NestedTensor_get_sizes(mat2_ptr),
      self_strides = NestedTensor_get_strides(self_ptr),
      mat2_strides = NestedTensor_get_strides(mat2_ptr);
  const std::vector<int64_t>& self_offsets = self_ptr->get_storage_offsets(),
      & mat2_offsets = mat2_ptr->get_storage_offsets();
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
  // to_padded_tensor only supports contiguous inputs
  auto self_contig = self.contiguous();
  auto mat2_contig = mat2.contiguous();
  // dispatcher should have guaranteed that at least one is nested
  const auto self_ptr = get_nested_tensor_impl(self_contig);
  const auto mat2_ptr = get_nested_tensor_impl(mat2_contig);
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
  TORCH_CHECK(self_dim == mat2_dim, "matmul: both inputs must have the same rank");
  int64_t ntensors = self_ptr->size(0),
      ntensors2 = mat2_ptr->size(0);
  TORCH_CHECK(ntensors == ntensors2,
      "matmul: Expected size for the 1st dimension of 2nd input tensor to be: ", ntensors,
      " but got: ", ntensors2, ".");
  // Ensure batch dimensions have the same sizes (no broadcasting).
  const auto& self_sizes = self_ptr->get_nested_size_tensor();
  const auto& mat2_sizes = mat2_ptr->get_nested_size_tensor();
  const auto& self_batch_sizes = self_sizes.narrow(1, 0, self_dim-3);
  const auto& mat2_batch_sizes = mat2_sizes.narrow(1, 0, mat2_dim-3);
  TORCH_CHECK(at::equal(self_batch_sizes, mat2_batch_sizes),
    "matmul: For nested tensors, batch dimensions must have the same sizes, ",
    "no broadcasting is currently performed. Got batch shapes for self ",
    self_batch_sizes,
    " and batch shapes for mat2 ",
    mat2_batch_sizes);
  // Ensure last dim of self and second last dim of mat2 have the same size
  const auto& self_dim_size = self_sizes.select(1, -1);
  const auto& mat2_dim_size = mat2_sizes.select(1, -2);
  TORCH_CHECK(at::equal(self_dim_size, mat2_dim_size),
    "matmul: Nested tensors cannot be matrix multiplied, last dimension of self has sizes",
    self_dim_size,
    "second last dimension of mat2 has sizes",
    mat2_dim_size);
  // Construct output size from input sizes
  Tensor output_sizes = self_sizes.clone();
  // The last entry in every row of output_sizes should be last column of mat2_sizes
  output_sizes.index_put_({at::indexing::Slice(), -1}, mat2_sizes.select(1, -1).clone());

  auto self_padded = self_contig.to_padded_tensor(0.);
  auto mat2_padded = mat2_contig.to_padded_tensor(0.);
  auto output_padded = at::matmul(self_padded, mat2_padded);
  auto output_nested = nested_from_padded_generic(output_padded, output_sizes);
  return output_nested;
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
      self, sizemat_transposed, stridemat_transposed, std::vector<int64_t>(self_ptr->get_storage_offsets()));
}

Tensor squeeze_nested(const Tensor& self) {
  TORCH_CHECK(false,
  "squeeze(): For nested tensors, squeeze without the dim argument is not supported ",
  "at the moment, however you can use squeeze(Tensor self, int dim) instead ",
  "if you need this feature, please open an issue on github describing your use case.");
  return self;
}

Tensor squeeze_dim_nested(const Tensor& self, int64_t dim) {
  auto self_ptr = get_nested_tensor_impl(self);
  int64_t ndim = self_ptr->dim();
  int64_t wrapped_dim = at::maybe_wrap_dim(dim, ndim);
  TORCH_CHECK(wrapped_dim > 0,
  "squeeze(): For nested tensors, squeezing dimension 0 is not supported at the moment ",
  "if you need this feature, please open an issue on github describing your use case.");
  const Tensor& sizemat = self_ptr->get_nested_size_tensor();
  const Tensor& stridemat = self_ptr->get_nested_stride_tensor();
  // if tensor.size(dim) != 1 torch.squeeze will return the result, we do the same here
  c10::optional<int64_t> size_dim = self_ptr->opt_size(dim);
  if (!(size_dim.has_value() && size_dim.value() == 1)) {
    // detach to avoid triggering throw_error_if_base_and_tensor_are_same
    return self.detach();
  }
  // if ndim == 2 and we pass the above if statement we should have a
  // nested tensor of singleton tensors
  TORCH_CHECK(ndim != 2,
  "squeeze(): For nested tensors, squeezing a nested tensor of singleton tensors is not ",
  "supported at the moment, if you need this feature, please open an issue on github",
  "describing your use case.");
  auto column_indices = sizemat.new_empty(ndim - 2);
  int64_t* column_indices_ptr = column_indices.data_ptr<int64_t>();
  std::iota(column_indices_ptr, column_indices_ptr + wrapped_dim - 1, 0);
  std::iota(column_indices_ptr + wrapped_dim - 1, column_indices_ptr + ndim - 2, wrapped_dim);
  auto sizemat_squeezed = at::index_select(sizemat, 1, column_indices);
  auto stridemat_squeezed = at::index_select(stridemat, 1, column_indices);
  return create_nested_view_tensor(
      self, sizemat_squeezed, stridemat_squeezed, std::vector<int64_t>(self_ptr->get_storage_offsets()));
}

Tensor unsqueeze_nested(const Tensor& self, int64_t dim) {
  auto self_ptr = get_nested_tensor_impl(self);
  int64_t ndim = self_ptr->dim();
  int64_t wrapped_dim = at::maybe_wrap_dim(dim, ndim + 1);
  TORCH_CHECK(wrapped_dim > 0,
  "unsqueeze(): For nested tensors, unsqueezing dimension 0 is not supported at the moment ",
  "if you need this feature, please open an issue on github describing your use case.");
  const Tensor& sizemat = self_ptr->get_nested_size_tensor();
  const Tensor& stridemat = self_ptr->get_nested_stride_tensor();
  auto mat_dim = wrapped_dim - 1;
  Tensor new_size = sizemat.new_ones({sizemat.size(0), 1});
  Tensor sizemat_unsqueezed = at::cat({sizemat.slice(1, 0, mat_dim),
                                       new_size,
                                       sizemat.slice(1, mat_dim, ndim)}, 1);
  Tensor new_stride;
  if (wrapped_dim == ndim) {
    new_stride = stridemat.new_ones({stridemat.size(0), 1});
  } else {
    new_stride = (stridemat.select(1, mat_dim - 1) * sizemat.select(1, mat_dim - 1)).unsqueeze(-1);
  }
  Tensor stridemat_unsqueezed = at::cat({stridemat.slice(1, 0, mat_dim),
                                         new_stride,
                                         stridemat.slice(1, mat_dim, ndim)}, 1);
  return create_nested_view_tensor(
      self, sizemat_unsqueezed, stridemat_unsqueezed, std::vector<int64_t>(self_ptr->get_storage_offsets()));
}


// utilities supporting `view_nested` and `reshape_nested`
namespace {
// Args:
//     sizes: the sizes of original nested tensor
//     strides: the strides of original nested tensor
//     proposed_shape: user proposed new shape
//     op: the options for new size and stride matrices
// Returns:
//     whether viewable
//     size matrix after reshape
//     stride matrix after reshape (not fully populated if not viewable)
inline std::tuple<bool, Tensor, Tensor> NestedTensor_compute_size_stride(
    const std::vector<IntArrayRef>& sizes,
    const std::vector<IntArrayRef>& strides,
    const IntArrayRef& proposed_shape,
    const c10::TensorOptions& op) {
  int64_t ntensors = sizes.size(),
      ndims_underlying = sizes[0].size(),
      ndims_underlying_reshaped = proposed_shape.size() - 1;
  bool viewable = true;
  Tensor sizemat_reshaped = at::empty({ntensors, ndims_underlying_reshaped}, op),
      stridemat_reshaped = at::empty({ntensors, ndims_underlying_reshaped}, op);
  int64_t* sizemat_reshaped_ptr = sizemat_reshaped.data_ptr<int64_t>(),
      * stridemat_reshaped_ptr = stridemat_reshaped.data_ptr<int64_t>();
  for (int64_t itensor = 0; itensor < ntensors; itensor++) {
    const IntArrayRef& size = sizes[itensor],
        & stride = strides[itensor];
    // compute reshaped size
    std::vector<int64_t> size_reshaped_vector(proposed_shape.begin() + 1, proposed_shape.end());
    // only allow one pre-existing dimension to have proposed shape == -1
    int64_t infer_index_old = -1;
    // some negative sizes remain to be infered
    if (ndims_underlying < ndims_underlying_reshaped) {
      int64_t numel = 1, numel_reshaped = 1;
      // replace negative sizes for old dimensions with old sizes
      for (int64_t idim = 0; idim < ndims_underlying; idim++) {
        int64_t& size_reshaped = size_reshaped_vector[idim];
        TORCH_CHECK(size_reshaped >= -1, "invalid shape dimension ", size_reshaped);
        if (size_reshaped == -1) {
          TORCH_CHECK(infer_index_old == -1, "only one dimension can be inferred");
          size_reshaped = size[idim];
          infer_index_old = idim;
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
      // See Note [Special size rule for nested tensor]
      TORCH_CHECK(infer_index == -1, "nested tensor does not infer shape");
      TORCH_CHECK(
          numel == numel_reshaped,
          "shape '", proposed_shape, "' ",
          "is invalid for input of size ", numel);
    }
    // all negative sizes can be replaced
    else {
      int64_t numel = 1, numel_reshaped = 1;
      for (int64_t idim = 0; idim < ndims_underlying_reshaped; idim++) {
        int64_t& size_reshaped = size_reshaped_vector[idim];
        TORCH_CHECK(size_reshaped >= -1, "invalid shape dimension ", size_reshaped);
        if (size_reshaped == -1) {
          size_reshaped = size[idim];
        }
        numel *= size[idim];
        numel_reshaped *= size_reshaped;
      }
      for (int64_t idim = ndims_underlying_reshaped; idim < ndims_underlying; idim++) {
        numel *= size[idim];
      }
      TORCH_CHECK(
          numel == numel_reshaped,
          "shape '", proposed_shape, "' ",
          "is invalid for input of size ", numel);
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
      viewable = false;
      // fill reshaped size into sizemat
      for (int64_t idim = 0; idim < ndims_underlying_reshaped; idim++) {
        sizemat_reshaped_ptr[idim] = size_reshaped[idim];
      }
      sizemat_reshaped_ptr += ndims_underlying_reshaped;
    }
  }
  return std::make_tuple(viewable, sizemat_reshaped, stridemat_reshaped);
}
} // namespace

// Note [Special size rule for nested tensor]
// Instead of infering size, -1 means "inherit the old size", so:
// * negative size is legal for a ragged dimension
// * however, we only allow one -1
// In principle we could still infer a dimension,
// we are designing a better semantics to include both inheritance and inference
Tensor view_nested(const Tensor& self, IntArrayRef proposed_shape) {
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
  int64_t ntensors_reshaped = proposed_shape[0];
  TORCH_CHECK(
      ntensors == ntensors_reshaped,
      "view: For now nested view cannot change or infer the implicit batch dimension");
  std::vector<IntArrayRef> sizes = NestedTensor_get_sizes(self_ptr),
      strides = NestedTensor_get_strides(self_ptr);
  // reshaping underlying tensor dimensions does not change offset
  // determine reshaped size and stride
  const Tensor& sizemat = self_ptr->get_nested_size_tensor();
  bool viewable;
  Tensor sizemat_reshaped, stridemat_reshaped;
  std::tie(viewable, sizemat_reshaped, stridemat_reshaped) = NestedTensor_compute_size_stride(
      sizes, strides, proposed_shape, sizemat.options());
  TORCH_CHECK(
      viewable,
      "view size is not compatible with input tensor's size and stride "
      "(at least one dimension spans across two contiguous subspaces). "
      "Use .reshape(...) instead.");
  return create_nested_view_tensor(self, sizemat_reshaped, stridemat_reshaped, std::vector<int64_t>(self_ptr->get_storage_offsets()));
}
  /**
   * Create a buffer tensor that is a view of self
   *
   * This serves as the boundary between nested and non nested tensor
   * view conversions
   *
   * @return Returns a new non nested tensor that
   * aliases the same storage as self
   */
Tensor values_nested(const Tensor& self) {
  TORCH_INTERNAL_ASSERT(self.is_nested(), "Can only create a buffer from Nested Tensor");
  auto* nt_self = get_nested_tensor_impl(self);
  return nt_self->get_unsafe_storage_as_tensor();
}

/**
 * Create a nested tensor that is a view of a buffer
 *
 * This serves as the boundary between non nested tensor and nested
 * view conversions
 *
 * @return Returns a nested tensor that
 * aliases the same storage as buffer
 */
Tensor _nested_view_from_buffer(
    const Tensor& buffer,
    const Tensor& nested_size_tensor,
    const Tensor& nested_stride_tensor,
    IntArrayRef offsets) {
  TORCH_INTERNAL_ASSERT(
      !buffer.is_nested(),
      "Can only a create Nested Tensor from a normal tensor buffer");
  TORCH_INTERNAL_ASSERT(buffer.dim() == 1, "The input buffer must be flat");
  TORCH_INTERNAL_ASSERT(nested_size_tensor.dim() == 2, "Expected the nested size tensor to be two dimensional.");
  uint64_t num_elements_nested_size = at::prod(nested_size_tensor, 1).sum().item<int64_t>();
  uint64_t buffer_storage_size = buffer.storage().nbytes()/buffer.dtype().itemsize();
  TORCH_INTERNAL_ASSERT(
      buffer_storage_size == num_elements_nested_size,
      "The number of elements in the buffer must equal the nested tensor size but buffer size: ",
      buffer_storage_size,
      " and nested tensor size: ",
      num_elements_nested_size,
      ".");

  TORCH_INTERNAL_ASSERT(nested_stride_tensor.dim() == 2, "Expected the nested stride tensor to be two dimensional.");
  TORCH_INTERNAL_ASSERT(nested_size_tensor.size(0) == nested_stride_tensor.size(0), "Expected the first dimension of nested size and nested stride tensor to be equal.");
  TORCH_INTERNAL_ASSERT(nested_stride_tensor.size(0) == (int64_t)offsets.size(), "Expected the first dimension of nested stride tensor to equal the length of offsets.");
  return at::detail::make_tensor<NestedTensorImpl>(
    c10::TensorImpl::VIEW,
    buffer,
    nested_size_tensor,
    nested_stride_tensor,
    std::vector<int64_t>(offsets.begin(), offsets.end()));
}

// See Note [Special size rule for nested tensor]
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
  int64_t ntensors_reshaped = proposed_shape[0];
  TORCH_CHECK(
      ntensors == ntensors_reshaped,
      "reshape: For now nested reshape cannot change or infer the implicit batch dimension");
  std::vector<IntArrayRef> sizes = NestedTensor_get_sizes(self_ptr),
      strides = NestedTensor_get_strides(self_ptr);
  // reshaping underlying tensor dimensions does not change offset
  // determine reshaped size and stride
  const Tensor& sizemat = self_ptr->get_nested_size_tensor();
  bool viewable{false};
  Tensor sizemat_reshaped, stridemat_reshaped;
  std::tie(viewable, sizemat_reshaped, stridemat_reshaped) = NestedTensor_compute_size_stride(
      sizes, strides, proposed_shape, sizemat.options());
  if (viewable) {
    return self.view(proposed_shape);
  }
  else {
    return self.clone(at::MemoryFormat::Contiguous).view(proposed_shape);
  }
}

Tensor reshape_as_nested(const Tensor& self, const Tensor& other) {
  auto other_ptr = get_nested_tensor_impl(other);
  // TODO: this is to reproduce other_ptr->opt_sizes_
  //       if an accessor is provided in the future, can replace this
  std::vector<int64_t> sizes;
  for (int64_t i = 0; i < other_ptr->dim(); i++) {
    c10::optional<int64_t> opt_size = other_ptr->opt_size(i);
    if (opt_size.has_value()) {
      sizes.push_back(*opt_size);
    }
    else {
      sizes.push_back(-1);
    }
  }
  // reshape with other.opt_sizes_
  return self.reshape(sizes);
}

} // namespace native
} // namespace at
