#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/layer_norm.h>
#include <ATen/NestedTensorImpl.h>
#include <c10/core/DispatchKey.h>

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
  TORCH_CHECK(buffer.is_contiguous(), "Given buffer must be contiguous.");
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(buffer), std::move(nested_size_tensor));
}

inline const at::Tensor& get_buffer(const at::Tensor& tensor) {
  return get_nested_tensor_impl(tensor)->get_buffer();
}

inline const at::Tensor& get_nested_size_tensor(const at::Tensor& tensor) {
  return get_nested_tensor_impl(tensor)->get_nested_size_tensor();
}

// CPU only!
// TODO: The algorithm here can be optimized, right now it involves a lot of
// small tensor manipulations
std::vector<at::Tensor> NestedTensor_unbind(
    const at::Tensor& self,
    int64_t dim) {
  TORCH_CHECK(
      dim == 0,
      "NestedTensor can only be unbound along dimension 0 ",
      "got dimension ",
      dim,
      " instead.");
  auto esizes = get_nested_size_tensor(self);
  std::vector<at::Tensor> result_tensors;
  if (esizes.dim() == 0) {
    return result_tensors;
  }
  auto esizes_chunks = esizes.unbind(0);
  std::vector<int64_t> splits;
  for (const auto i : c10::irange(esizes_chunks.size())) {
    splits.push_back(esizes_chunks[i].prod().item<int64_t>());
  }
  auto buffer_chunks = at::split_with_sizes(get_buffer(self), splits);
  for (const auto i : c10::irange(buffer_chunks.size())) {
    const auto& esize_chunk = esizes_chunks[i];
    result_tensors.push_back(buffer_chunks[i].view(IntArrayRef(
        esize_chunk.data_ptr<int64_t>(),
        esize_chunk.data_ptr<int64_t>() + esize_chunk.numel())));
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
  const auto& sizes = nt.get_nested_size_tensor();
  return NestedTensor_get_max_size_from_size_tensor(sizes);
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
  const auto target_size = NestedTensor_get_max_size_from_size_tensor(sizes);
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
  at::Tensor new_buffer = padded_transformed.masked_select(final_mask);
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(new_buffer), sizes);
}

Tensor NestedTensor_to_padded_tensor(const Tensor& t, double padding) {
  // TODO port CUDA path in pytorch/nestedtensor to_padded_tensor!
  // TODO: skipped optimization for case of all 1x1 tensors
  auto& nt = *get_nested_tensor_impl(t);
  auto max_size = NestedTensor_get_max_size(nt);
  auto sizes = nt.get_nested_size_tensor();

  if (sizes.numel() == 0 || sizes.dim() == 0) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(nt.get_buffer().numel() == 0);
    return nt.get_buffer();
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
  return at::stack(buffers);
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
} // namespace native
} // namespace at
