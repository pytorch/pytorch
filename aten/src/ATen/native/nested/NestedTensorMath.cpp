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
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/Resize.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#include <tuple>
#include <utility>


namespace at::native {
namespace {

int64_t num_bytes(IntArrayRef sizes) {
  // 0-dim Tensors have torch.Size of .size() 0, but carry 1 memory.
  // Empty 1-dim Tensors (torch.tensor([])) have torch.Size of .size() 1,
  // but carry 0 memory.
  int64_t result = 1;
  int64_t stride = 1;
  for (int64_t ii = static_cast<int64_t>(sizes.size()) - 1; ii >= 0; --ii) {
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
  for (int64_t i = static_cast<int64_t>(tup.size()) - 1; i >= 0; i--) {
    padd.push_back(0);
    padd.push_back(goal_shape[i] - tup[i]);
  }
  Tensor new_tensor = at::constant_pad_nd(t, IntArrayRef(padd), value);
  new_tensor = new_tensor.reshape(goal_shape);
  return new_tensor;
}
} // namespace


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
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
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

std::tuple<Tensor, Tensor, Tensor> nested_layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps) {
  TORCH_CHECK_VALUE(weight_opt && bias_opt, "NestedTensor layer_norm requires weight and bias");
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
      std::nullopt /* dtype */,
      std::nullopt /* layout */,
      std::nullopt /* device */,
      std::nullopt /* pin_memory */,
      at::MemoryFormat::Contiguous);
  auto options = input_buffer.options();
  if (input_buffer.is_cuda() || input_buffer.is_xpu()) {
    auto acc_type = at::toAccumulateType(input_buffer.scalar_type(), input_buffer.device().type());
    options = options.dtype(acc_type);
  }
  Tensor mean = at::empty({M}, options);
  Tensor rstd = at::empty({M}, options);
  LayerNormKernel(
      input_buffer.device().type(),
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
    wrap_buffer(output_buffer, nt_input->get_nested_sizes()),
    mean,
    rstd
  );
}

Tensor NestedTensor_from_padded_and_nested_example(
    const Tensor& padded,
    const Tensor& nt_example) {
  return _nested_from_padded(padded, get_nested_tensor_impl(nt_example)->get_nested_sizes());
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
        sizes_i, true, kBool, std::nullopt, std::nullopt, std::nullopt);
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
  auto sizes = nt.get_nested_sizes();

  if (sizes.numel() == 0 || sizes.dim() == 0) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(nt.get_buffer().numel() == 0);
    return nt.get_buffer().clone();
  }
  TORCH_CHECK(
      t.numel() > 0,
      "to_padded_tensor: at least one constituent tensor should have non-zero numel"
  )

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
    for (int64_t i = 0; i < ret_val.dim(); i++) {
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
  const auto& sizes = nt_indices->get_nested_sizes();
  auto new_sizes = at::empty({sizes.size(0)}, sizes.options());
  new_sizes.fill_(weight.sizes()[1]);
  new_sizes = new_sizes.reshape({new_sizes.size(0), 1});
  new_sizes = at::cat({sizes, new_sizes}, 1);
  return at::detail::make_tensor<NestedTensorImpl>(
      result_buffer.reshape({-1}), std::move(new_sizes));
}

// Very rudimentary sum_dim for prototyping with torch_scatter.segment_reduce.
Tensor NestedTensor_sum_dim_CPU(
    const Tensor& self,
    OptionalIntArrayRef opt_dims,
    bool keepdim,
    std::optional<ScalarType> dtype) {
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

  auto sizemat = nt_input->get_nested_sizes();
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
    const auto* input_data = buffer.const_data_ptr<scalar_t>();
    int64_t out_idx = 0, in_idx = 0;
    for (const auto i : c10::irange(ntensors)) {
      int64_t segments = num_segments[i].item<int64_t>();
      int64_t segment_length = segment_lengths[i].item<int64_t>();
      for (int64_t j = 0; j < segments; j++) {
        scalar_t res = 0;
        for (int64_t k = 0; k < segment_length; k++) {
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
  std::vector<IntArrayRef> sizes = NestedTensor_get_sizes(self_ptr),
                           strides = NestedTensor_get_strides(self_ptr);
  int64_t *offsets_ptr = self_ptr->get_storage_offsets().data_ptr<int64_t>();
  const at::Tensor& buffer = self_ptr->get_unsafe_storage_as_tensor();
  int64_t positive_dim = at::maybe_wrap_dim(dim, self_ptr->dim());
  int64_t ntensors = self_ptr->size(0);
  TORCH_CHECK_INDEX(ntensors > 0, "You can only select when the NT is not empty.");
  int64_t ndims = static_cast<long>(sizes[0].size());
  if (positive_dim == 0) {
    TORCH_CHECK_INDEX(
        index >= -ntensors && index < ntensors,
        "index ",
        index,
        " is out of bounds for dimension 0 with size ",
        ntensors);
    int64_t positive_index = index < 0 ? index + ntensors : index;
    return buffer.as_strided(
        sizes[positive_index],
        strides[positive_index],
        offsets_ptr[positive_index]);
  } else {
    auto new_sizes = at::empty({ntensors, ndims-1}, TensorOptions().dtype(kLong));
    auto new_strides = at::empty({ntensors, ndims-1}, TensorOptions().dtype(kLong));
    auto new_offsets = at::empty({ntensors}, TensorOptions().dtype(kLong));
    for (int64_t i : c10::irange(ntensors)) {
      int64_t *size_ptr = new_sizes[i].data_ptr<int64_t>();
      int64_t *stride_ptr = new_strides[i].data_ptr<int64_t>();

      int64_t dim_idx = 0;
      for (int64_t j : c10::irange(ndims)) {
        if (j != dim - 1) {
          size_ptr[dim_idx] = sizes[i][j];
          stride_ptr[dim_idx] = strides[i][j];
          ++dim_idx;
        } else {
          TORCH_CHECK_INDEX(
              index >= 0 && index < sizes[i][j],
              "index ",
              index,
              " is out of bounds for dimension ",
              j,
              " of the ",
              i,
              "th constituent tensor with size ",
              sizes[i][j]);
          new_offsets[i] = offsets_ptr[i] + index * strides[i][j];
        }
      }
    }
    return create_nested_view_tensor(self, new_sizes, new_strides, new_offsets);
  }

}

std::tuple<Tensor,Tensor> native_dropout_nested(const Tensor& input, double p, std::optional<bool> train) {
  auto input_ptr = get_nested_tensor_impl(input);
  const Tensor& input_buffer = input_ptr-> get_unsafe_storage_as_tensor(),
      & sizemat = input_ptr->get_nested_sizes(),
      & stridemat = input_ptr->get_nested_strides();
  const auto offsets = input_ptr->get_storage_offsets();
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
  Tensor output = wrap_buffer(output_buffer, sizemat.clone(), stridemat.clone(), offsets.clone()),
      mask = wrap_buffer(mask_buffer, sizemat.clone(), stridemat.clone(), offsets.clone());
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
      & sizemat = input_ptr->get_nested_sizes();
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

Tensor NestedTensor_all(
    const Tensor& input,
    const int64_t dim,
    const bool keepdim) {
  auto input_ptr = get_nested_tensor_impl(input);
  int64_t ntensors = input_ptr->size(0);
  if (ntensors == 0) {
    return input.clone();
  }
  int64_t positive_dim = at::maybe_wrap_dim(dim, input_ptr->dim());
  TORCH_CHECK(
      positive_dim >= 1,
      "Cannot apply all across nested dimension 0");
  const Tensor& buffer = input_ptr->get_unsafe_storage_as_tensor(),
      & sizemat = input_ptr->get_nested_sizes();


  Tensor output_buffer = buffer.new_empty(buffer.sizes());

  Tensor output_size = sizemat.clone();
  if (keepdim) {
    output_size.select(1, positive_dim - 1).fill_(1);
  } else {
    output_size = output_size.slice(1, 0, positive_dim - 1);
  }

  Tensor output = wrap_buffer(output_buffer, output_size.contiguous());

  std::vector<Tensor> input_unbind = input.unbind(),
      output_unbind = output.unbind();
  for (int64_t i = 0; i < ntensors; i++) {
    at::all_out(
        output_unbind[i],
        input_unbind[i],
        positive_dim - 1,
        keepdim);
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
  const Tensor& sizemat = self_ptr->get_nested_sizes(),
      & stridemat = self_ptr->get_nested_strides();
  Tensor column_indices = sizemat.new_empty(ndims);
  int64_t* column_indices_ptr = column_indices.data_ptr<int64_t>();
  std::iota(column_indices_ptr, column_indices_ptr + ndims, 0);
  column_indices_ptr[positive_dim0] = positive_dim1;
  column_indices_ptr[positive_dim1] = positive_dim0;
  // create transposed `sizemat` and `stridemat`
  Tensor sizemat_transposed = at::index_select(sizemat, 1, column_indices),
      stridemat_transposed = at::index_select(stridemat, 1, column_indices);
  return create_nested_view_tensor(
      self, sizemat_transposed, stridemat_transposed, self_ptr->get_storage_offsets().clone());
}

Tensor squeeze_nested(const Tensor& self) {
  TORCH_CHECK(false,
  "squeeze(): For nested tensors, squeeze without the dim argument is not supported ",
  "at the moment, however you can use squeeze(Tensor self, int dim) instead ",
  "if you need this feature, please open an issue on github describing your use case.");
  return self;
}

Tensor squeeze_dim_nested(const Tensor& self, IntArrayRef dims) {
  auto self_ptr = get_nested_tensor_impl(self);
  int64_t ndim = self_ptr->dim();
  auto mask = at::dim_list_to_bitset(dims, ndim);
  TORCH_CHECK(!mask.test(0),
  "squeeze(): For nested tensors, squeezing dimension 0 is not supported at the moment ",
  "if you need this feature, please open an issue on github describing your use case.");
  const Tensor& sizemat = self_ptr->get_nested_sizes();
  const Tensor& stridemat = self_ptr->get_nested_strides();
  // if tensor.size(dim) != 1 torch.squeeze will return the result, we do the same here
  for (const auto d : c10::irange(ndim)) {
    if (mask.test(d)) {
      std::optional<int64_t> size_dim = self_ptr->opt_size(d);
      if (!(size_dim.has_value() && *size_dim == 1)) {
        mask.reset(d);
      }
    }
  }

  if (!mask.any()) {
    // detach to avoid triggering throw_error_if_base_and_tensor_are_same
    return self.detach();
  }
  // if ndim == 2 and we pass the above if statement we should have a
  // nested tensor of singleton tensors
  TORCH_CHECK(ndim > static_cast<int64_t>(1 + dims.size()),
  "squeeze(): For nested tensors, squeezing a nested tensor of singleton tensors is not ",
  "supported at the moment, if you need this feature, please open an issue on github",
  "describing your use case.");
  const auto new_ndim = ndim - mask.count();
  auto column_indices = sizemat.new_empty(static_cast<int64_t>(new_ndim) - 1);
  int64_t* column_indices_ptr = column_indices.data_ptr<int64_t>();
  for (const auto d : c10::irange(1, ndim)) {
    if (!mask.test(d)) {
      *column_indices_ptr++ = d - 1;
    }
  }
  auto sizemat_squeezed = at::index_select(sizemat, 1, column_indices);
  auto stridemat_squeezed = at::index_select(stridemat, 1, column_indices);
  return create_nested_view_tensor(
      self, sizemat_squeezed, stridemat_squeezed, self_ptr->get_storage_offsets().clone());
}

Tensor squeeze_dim_nested(const Tensor& self, int64_t dim) {
  return squeeze_dim_nested(self, IntArrayRef{dim});
}

Tensor unsqueeze_nested(const Tensor& self, int64_t dim) {
  auto self_ptr = get_nested_tensor_impl(self);
  int64_t ndim = self_ptr->dim();
  int64_t wrapped_dim = at::maybe_wrap_dim(dim, ndim + 1);
  TORCH_CHECK(wrapped_dim > 0,
  "unsqueeze(): For nested tensors, unsqueezing dimension 0 is not supported at the moment ",
  "if you need this feature, please open an issue on github describing your use case.");
  const Tensor& sizemat = self_ptr->get_nested_sizes();
  const Tensor& stridemat = self_ptr->get_nested_strides();
  auto mat_dim = wrapped_dim - 1;
  Tensor new_size = sizemat.new_ones({sizemat.size(0), 1});
  Tensor sizemat_unsqueezed = at::cat({sizemat.slice(1, 0, mat_dim),
                                       new_size,
                                       sizemat.slice(1, mat_dim, ndim)}, 1);
  Tensor new_stride;
  if (wrapped_dim == ndim) {
    new_stride = stridemat.new_ones({stridemat.size(0), 1});
  } else {
    new_stride = (stridemat.select(1, mat_dim) * sizemat.select(1, mat_dim)).unsqueeze(-1);
  }
  Tensor stridemat_unsqueezed = at::cat({stridemat.slice(1, 0, mat_dim),
                                         new_stride,
                                         stridemat.slice(1, mat_dim, ndim)}, 1);
  return create_nested_view_tensor(
      self, sizemat_unsqueezed, stridemat_unsqueezed, self_ptr->get_storage_offsets().clone());
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
  int64_t ntensors = static_cast<int64_t>(sizes.size());
  int64_t ndims_underlying = static_cast<int64_t>(sizes[0].size());
  int64_t ndims_underlying_reshaped = static_cast<int64_t>(proposed_shape.size() - 1);
  bool viewable = true;
  Tensor sizemat_reshaped = at::empty({ntensors, ndims_underlying_reshaped}, op),
      stridemat_reshaped = at::empty({ntensors, ndims_underlying_reshaped}, op);
  int64_t* sizemat_reshaped_ptr = sizemat_reshaped.mutable_data_ptr<int64_t>(),
      * stridemat_reshaped_ptr = stridemat_reshaped.mutable_data_ptr<int64_t>();
  for (int64_t itensor = 0; itensor < ntensors; itensor++) {
    const IntArrayRef& size = sizes[itensor],
        & stride = strides[itensor];
    // compute reshaped size
    std::vector<int64_t> size_reshaped_vector(proposed_shape.begin() + 1, proposed_shape.end());
    // only allow one pre-existing dimension to have proposed shape == -1
    int64_t infer_index_old = -1;
    // some negative sizes remain to be inferred
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
          TORCH_CHECK(false, "invalid shape dimension ", size_reshaped);
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
// Instead of inferring size, -1 means "inherit the old size", so:
// * negative size is legal for a ragged dimension
// * however, we only allow one -1
// In principle we could still infer a dimension,
// we are designing a better semantics to include both inheritance and inference
Tensor view_nested(const Tensor& self, IntArrayRef proposed_shape) {
  TORCH_CHECK(
      !proposed_shape.empty(),
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
  const Tensor& sizemat = self_ptr->get_nested_sizes();
  auto [viewable, sizemat_reshaped, stridemat_reshaped] = NestedTensor_compute_size_stride(
      sizes, strides, proposed_shape, sizemat.options());
  TORCH_CHECK(
      viewable,
      "view size is not compatible with input tensor's size and stride "
      "(at least one dimension spans across two contiguous subspaces). "
      "Use .reshape(...) instead.");
  return create_nested_view_tensor(self, sizemat_reshaped, stridemat_reshaped, self_ptr->get_storage_offsets().clone());
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
    const Tensor& nested_sizes,
    const Tensor& nested_strides,
    const Tensor& storage_offsets) {
  TORCH_INTERNAL_ASSERT(
      !buffer.is_nested(),
      "Can only a create Nested Tensor from a normal tensor buffer");
  TORCH_INTERNAL_ASSERT(buffer.dim() == 1, "The input buffer must be flat");
  TORCH_INTERNAL_ASSERT(nested_sizes.dim() == 2, "Expected the nested size tensor to be two dimensional.");
  TORCH_INTERNAL_ASSERT(nested_strides.dim() == 2, "Expected the nested stride tensor to be two dimensional.");
  TORCH_INTERNAL_ASSERT(nested_sizes.size(0) == nested_strides.size(0), "Expected the first dimension of nested size and nested stride tensor to be equal.");
  TORCH_INTERNAL_ASSERT(nested_strides.size(0) == storage_offsets.size(0), "Expected the first dimension of nested stride tensor to equal the length of offsets.");


  std::vector<at::Tensor> all_sizes = nested_sizes.unbind();
  std::vector<at::Tensor> all_strides = nested_strides.unbind();
  std::vector<at::Tensor> all_offsets = storage_offsets.unbind();
  auto size_dim = nested_sizes.size(1);

  for (const auto i : c10::irange(nested_sizes.size(0))) {
    const int64_t* sizemat_ptr = all_sizes[i].const_data_ptr<int64_t>();
    const int64_t* stridemat_ptr = all_strides[i].const_data_ptr<int64_t>();
    const int64_t* offset_ptr = all_offsets[i].const_data_ptr<int64_t>();
    checkInBoundsForStorage(
        IntArrayRef(sizemat_ptr, sizemat_ptr + size_dim),
        IntArrayRef(stridemat_ptr, stridemat_ptr + size_dim),
        *offset_ptr,
        buffer.dtype(),
        buffer.storage());
  }

  return at::detail::make_tensor<NestedTensorImpl>(
    c10::TensorImpl::VIEW,
    buffer,
    nested_sizes,
    nested_strides,
    storage_offsets);
}

std::tuple<Tensor, Tensor> _nested_compute_contiguous_strides_offsets(const Tensor& nested_size) {
  return std::make_tuple(
      construct_nested_strides(nested_size),
      construct_offsets(nested_size));
}

// See Note [Special size rule for nested tensor]
Tensor reshape_nested(const Tensor& self, IntArrayRef proposed_shape) {
  TORCH_CHECK(
      !proposed_shape.empty(),
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
  const Tensor& sizemat = self_ptr->get_nested_sizes();
  auto [viewable, sizemat_reshaped, stridemat_reshaped] = NestedTensor_compute_size_stride(
      sizes, strides, proposed_shape, sizemat.options());
  if (viewable) {
    return self.view(proposed_shape);
  }
  else {
    return self.clone(at::MemoryFormat::Contiguous).view(proposed_shape);
  }
}

Tensor reshape_nested_symint(const Tensor& self, SymIntArrayRef proposed_shape) {
  // Jagged layout NT decomp
  if (self.layout() == at::kJagged) {
    // TODO: Expand decomp to handle other viewable cases
    bool viewable = self.is_contiguous();
    return (
        viewable ? self.view_symint(proposed_shape) :
        self.clone(at::MemoryFormat::Contiguous).view_symint(proposed_shape)
    );
  }

  return reshape_nested(self, C10_AS_INTARRAYREF_SLOW(proposed_shape));
}

Tensor reshape_as_nested(const Tensor& self, const Tensor& other) {
  // Jagged layout NT decomp
  if (self.layout() == at::kJagged) {
    return self.reshape_symint(other.sym_sizes());
  }

  auto other_ptr = get_nested_tensor_impl(other);
  // TODO: this is to reproduce other_ptr->opt_sizes_
  //       if an accessor is provided in the future, can replace this
  std::vector<int64_t> sizes;
  for (int64_t i = 0; i < other_ptr->dim(); i++) {
    std::optional<int64_t> opt_size = other_ptr->opt_size(i);
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

Tensor& normal_nested_(Tensor& self, double mean, double std, std::optional<Generator> gen) {
  const auto& self_buf = get_nested_tensor_impl(self)->get_buffer();
  self_buf.normal_(mean, std, std::move(gen));
  return self;
}

// returns true if the sizes are compatible to be concatenated along the specified dim
// sizes should match outside of the dim of concatenation
static bool can_cat_nested_sizes(const Tensor& nested_sizes1, const Tensor& nested_sizes2, int64_t cat_dim) {
  if (nested_sizes1.sizes() != nested_sizes2.sizes()) {
    return false;
  }

  auto nested_sizes1_ptr = nested_sizes1.data_ptr<int64_t>();
  auto nested_sizes2_ptr = nested_sizes2.data_ptr<int64_t>();
  const auto num_components = nested_sizes1.size(0);
  const auto num_dims = nested_sizes1.size(1);
  for (auto c : c10::irange(num_components)) {
    for (auto d : c10::irange(num_dims)) {
      // subtract 1 to account for batch dim
      auto component_cat_dim = cat_dim - 1;
      if (d == component_cat_dim) {
        continue;
      }
      if (nested_sizes1_ptr[c * num_dims + d] != nested_sizes2_ptr[c * num_dims + d]) {
        return false;
      }
    }
  }

  return true;
}

// cat a list of NTs that are representable as jagged
static Tensor cat_nested_as_jagged(
    const MaterializedITensorListRef& tensors,
    int64_t dim) {
  const auto first_item = tensors[0].get();
  const auto first_item_dim = first_item.dim();
  const auto first_item_batch_size = first_item.size(0);
  std::vector<Tensor> jagged_views;
  for (auto i : c10::irange(tensors.size())) {
    auto t = tensors[i].get();
    TORCH_CHECK(t.is_nested(),
        "cat(): expected each tensor in given list to be nested");
    TORCH_CHECK(t.is_contiguous(),
        "cat(): only contiguous nested tensors are supported");
    if (i > 0) {
      TORCH_CHECK(
          can_cat_nested_sizes(
              get_nested_tensor_impl(first_item)->get_nested_sizes(),
              get_nested_tensor_impl(t)->get_nested_sizes(),
              dim),
          "cat(): expected all nested tensors to have matching ragged structures outside of the concatenated dim");
    }
    // only support inputs in the form (B, *, D_0, D_1, ...)
    // i.e. require at most a single ragged dim next to the batch dim
    auto *nt_impl = get_nested_tensor_impl(t);
    std::vector<int64_t> jagged_size;
    jagged_size.push_back(-1);
    for (auto d : c10::irange(first_item_dim - 2)) {
      TORCH_CHECK(nt_impl->opt_size(d + 2).has_value(),
          "cat(): only nested tensors with a single ragged dim next to the batch dim are supported");
      jagged_size.push_back(nt_impl->size(d + 2));
    }
    auto jagged = nt_impl->get_buffer().view(jagged_size);
    jagged_views.push_back(jagged);
  }

  // view each of the NTs as jagged for the cat() call
  auto new_buffer = at::cat(jagged_views, dim - 1);

  // wrap result into nested tensor
  const auto component_dim = first_item_dim - 1;
  auto new_dim_size = new_buffer.size(dim - 1);
  auto new_sizes = get_nested_tensor_impl(tensors[0].get())->get_nested_sizes().clone();
  auto new_sizes_ptr = new_sizes.data_ptr<int64_t>();
  for (const auto i : c10::irange(first_item_batch_size)) {
    new_sizes_ptr[i * component_dim + (dim - 1)] = new_dim_size;
  }
  return at::detail::make_tensor<NestedTensorImpl>(
      new_buffer.view(-1), new_sizes);
}

static Tensor cat_nested_impl(
    const MaterializedITensorListRef& tensors,
    int64_t dim) {
  dim = maybe_wrap_dim(dim, tensors[0].get());
  if (dim == 0) {
    // handle simple case of dim=0: concat NT components
    std::vector<at::Tensor> buffers;
    std::vector<at::Tensor> sizes;
    for (const auto i : c10::irange(tensors.size())) {
      const Tensor& t = tensors[i];
      TORCH_CHECK(
          t.is_nested(), "Expected each tensor in given list to be nested.");
      TORCH_CHECK(
          t.is_contiguous(),
          "Expected each tensor in given list to be contiguous.");
      auto t_ptr = get_nested_tensor_impl(t);
      buffers.push_back(t_ptr->get_buffer().view({-1}));
      sizes.push_back(t_ptr->get_nested_sizes());
    }
    return at::detail::make_tensor<NestedTensorImpl>(
        at::cat(buffers).view({-1}), at::cat(sizes, 0));
  }

  // NB: support for other dims is restricted to nested tensors representable as jagged
  return cat_nested_as_jagged(tensors, dim);
}

Tensor cat_nested(const ITensorListRef& tensors, int64_t dim) {
  auto materialized = tensors.materialize();
  return cat_nested_impl(materialized, at::legacy_cat_wrap_dim(dim, materialized));
}

} // namespace at::native
