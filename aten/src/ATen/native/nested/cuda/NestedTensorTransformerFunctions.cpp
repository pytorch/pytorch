#include <numeric>
#include <algorithm>
#include <type_traits>
#include <c10/util/Exception.h>

#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/NonSymbolicBC.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_nested_from_padded.h>
#endif

#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
#include <ATen/native/nested/NestedTensorMath.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/native/transformers/cuda/sdp_utils.h>

#include <ATen/cuda/CUDAContext.h>
namespace at {
namespace native {
namespace {
int64_t padded_tensor_numel(const Tensor& sizes) {
  const auto sizes_num_rows = sizes.sizes()[0];
  const auto sizes_row_length = sizes.sizes()[1];
  const auto* sizes_data = sizes.data_ptr<int64_t>();
  int64_t numel = 0;
  for (const auto row_num : c10::irange(sizes_num_rows)) {
    const auto* row_ptr = sizes_data + row_num * sizes_row_length;
    int64_t prod = 1;
    for (const auto idx : c10::irange(sizes_row_length)) {
      prod *= row_ptr[idx];
    }
    numel += prod;
  }
  return numel;
}
} // namespace
Tensor nested_from_padded_cuda(
    const Tensor& padded,
    const Tensor& sizes,
    bool do_transform_0213) {
  if (padded.dim() > 1 && padded.dim() < 5) {
    // Instead of erroring call the generic version
    if(!(padded.dim() == 4 && do_transform_0213) && !(padded.dim() == 3 && !do_transform_0213)){
      return at::native::nested_from_padded_generic(padded, sizes, do_transform_0213);
    }
    if (padded.dtype() != kFloat && padded.dtype() != kHalf) {
      TORCH_WARN_ONCE(
          "nested_from_padded CUDA kernels only support fp32/fp16; falling "
          "back to slower generic kernel");
      return at::native::nested_from_padded_generic(padded, sizes, do_transform_0213);
    }
    Tensor target_offsets =
        NestedTensor_batch_offsets_from_size_tensor(sizes, 0);
    Tensor padded_sizes_tensor = at::tensor(padded.sizes());
    Tensor output = at::empty({padded_tensor_numel(sizes)}, padded.options());
    Tensor target_size_sizes = sizes.reshape(-1);

    Tensor metadata =
        at::cat({target_size_sizes, padded_sizes_tensor, target_offsets});
    metadata = metadata.to(at::Device(kCUDA), kInt, true, true);

    auto output_size_ptr = metadata.data_ptr<int>();
    auto input_size_ptr = output_size_ptr + target_size_sizes.numel();
    auto offsets_ptr = input_size_ptr + padded_sizes_tensor.numel();

    Tensor padded_contiguous = padded.contiguous();
    if (padded.dtype() == kFloat) {
      if (do_transform_0213) {
        remove_padding_transform0213_kernelLauncher(
            padded_contiguous.data_ptr<float>(),
            output.data_ptr<float>(),
            offsets_ptr,
            input_size_ptr,
            output_size_ptr,
            padded_contiguous.dim() - 2,
            padded_contiguous.sizes()[0]);
      } else {
        remove_padding_kernelLauncher(
            padded_contiguous.data_ptr<float>(),
            output.data_ptr<float>(),
            offsets_ptr,
            input_size_ptr,
            output_size_ptr,
            padded_contiguous.dim() - 1,
            padded_contiguous.sizes()[0]);
      }
    } else if (padded.dtype() == kHalf) {
      if (do_transform_0213) {
        remove_padding_transform0213_kernelLauncher(
            padded_contiguous.data_ptr<c10::Half>(),
            output.data_ptr<c10::Half>(),
            offsets_ptr,
            input_size_ptr,
            output_size_ptr,
            padded_contiguous.dim() - 2,
            padded_contiguous.sizes()[0]);
      } else {
        remove_padding_kernelLauncher(
            padded_contiguous.data_ptr<c10::Half>(),
            output.data_ptr<c10::Half>(),
            offsets_ptr,
            input_size_ptr,
            output_size_ptr,
            padded_contiguous.dim() - 1,
            padded_contiguous.sizes()[0]);
      }
    } else {
      AT_ERROR("Only support fp32/fp16 for padded input");
    }
    return at::detail::make_tensor<NestedTensorImpl>(std::move(output), sizes);
  } else {
    return at::native::nested_from_padded_generic(padded, sizes);
  }
}

Tensor batch_offsets_from_efficient_size(const Tensor& ef_sizes) {
  int64_t* nt_sizes_ptr = ef_sizes.data_ptr<int64_t>();
  int64_t ef_sizes_size_0 = ef_sizes.sizes()[0];
  Tensor offsets = at::empty({1 + ef_sizes_size_0}, at::kLong);
  int64_t* offsets_ptr = offsets.mutable_data_ptr<int64_t>();
  offsets_ptr[0] = 0;
  int64_t ef_sizes_size_1 = ef_sizes.sizes()[1];
  for (const auto i : c10::irange(ef_sizes_size_0)) {
    int64_t prod = 1;
    for (const auto j : c10::irange(ef_sizes_size_1)) {
      prod = prod * nt_sizes_ptr[i * ef_sizes_size_1 + j];
    }
    offsets_ptr[i + 1] = offsets_ptr[i] + prod;
  }
  return offsets;
}

Tensor NestedTensor_to_padded_tensor_cuda(
    const Tensor& t,
    double padding,
    OptionalIntArrayRef output_size) {
  int64_t t_dim = t.dim();
  if (t_dim >= 2 && t_dim <= 4 &&
      (t.dtype() == at::kFloat || t.dtype() == at::kDouble ||
       t.dtype() == at::kHalf)) {
    auto* nt_input = get_nested_tensor_impl(t);
    TORCH_CHECK(
        nested_tensor_impl_is_contiguous(nt_input),
        "for now to_padded_tensor only supports contiguous nested tensor");
    const auto& nt_buffer = nt_input->get_buffer();

    if (t_dim == 3 && nt_input->opt_size(2) && (*nt_input->opt_size(2) > 0) &&
        !(output_size.has_value())) {
      Tensor nt_sizes = nt_input->get_nested_sizes();
      Tensor sizes_dim1 = at::native::narrow_symint(nt_sizes, 1, 0, 1);
      Tensor sizes_dim2 = at::native::narrow_symint(nt_sizes, 1, 1, 1);
      Tensor result = at::detail::make_tensor<NestedTensorImpl>(
          nt_input->get_buffer(), sizes_dim1 * sizes_dim2[0]);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.dim() == 2);
      result =
          NestedTensor_to_padded_tensor_cuda(result, padding, output_size);
      return result.reshape({result.sizes()[0], -1, *nt_input->opt_size(2)});
    }

    Tensor nt_sizes = nt_input->get_nested_sizes();
    Tensor offsets = batch_offsets_from_efficient_size(nt_sizes);
    auto new_size = NestedTensor_get_max_size(*nt_input);
    new_size.insert(new_size.begin(), nt_sizes.sizes()[0]);

    // Pad output tensor to output_size if provided
    if (output_size.has_value()) {
      auto output_size_ = output_size.value();
      TORCH_CHECK(
          output_size_.size() == new_size.size(),
          "Length of output_size does not match NestedTensor dims. Broadcasting is not supported.");
      for (uint64_t i = 0; i < new_size.size(); i++) {
        TORCH_CHECK(
            output_size_[i] >= new_size[i],
            "Value in output_size is less than NestedTensor padded size. Truncation is not supported.");
        new_size[i] = output_size_[i];
      }
    }

    Tensor output = at::empty(IntArrayRef(new_size), nt_buffer.options());

    int64_t input_dim = nt_sizes.sizes()[1];
    int64_t batch_size = nt_sizes.sizes()[0];
    int64_t output_batch_size = new_size[0];
    // TODO: Remove need for cat here
    at::Tensor metadata = at::cat({offsets, nt_sizes.reshape(-1)});
    metadata = metadata.to(at::Device(kCUDA), at::kInt);

    std::vector<Tensor> split =
        at::split_with_sizes(metadata, {offsets.numel(), nt_sizes.numel()}, 0);

    offsets = split[0];
    nt_sizes = split[1];

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        nt_buffer.scalar_type(), "NestedTensor_to_padded_tensor_cuda", [&]() {
          add_padding_kernelLauncher(
              nt_buffer.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(),
              (scalar_t)(padding),
              offsets.data_ptr<int>(),
              nt_sizes.data_ptr<int>(),
              input_dim,
              new_size,
              batch_size,
              output_batch_size);
        });
    return output;
  }
  return NestedTensor_to_padded_tensor_generic(t, padding, output_size);
}

namespace{

/**
 * This function is used to calculate two pieces of metadata that are needed
 * for use with flash-attention and efficient_attention kernels. They are the
 * cumulative sequence_length over a batch of sequences and the maximum sequence
 * length.
 *
 * @return A tuple of cumulative sequence lengths and the maximum sequence length,
 * and the last element in the cumulative_sequence_lengths
 */
std::tuple<Tensor, int64_t, int64_t> cumulative_and_max_seq_len(Tensor qkv) {
  TORCH_CHECK(
      qkv.is_nested(),
      "QKV must be nested for flash cumulative_seq_len calculation.")
  auto* nt_impl = get_nested_tensor_impl(qkv);
  const auto& sizes = nt_impl->get_nested_sizes();
  auto size_tensor_stride = sizes.stride(0);

  const int64_t batch_size = qkv.size(0);
  auto cumulative_seqlen = at::zeros(
      {batch_size + 1}, TensorOptions().device(at::kCPU).dtype(at::kInt));

  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  auto* cumulative_seqlen_ptr = cumulative_seqlen.data_ptr<int32_t>();

  int32_t sum = 0;
  int64_t max_seqlen = -1;
  cumulative_seqlen_ptr[0] = sum;
  for (const auto i : c10::irange(batch_size)) {
    // Calculate the cumulative sum of the sequence lengths
    auto current_seq_len = sizes_ptr[(i * size_tensor_stride)];
    sum += current_seq_len;
    cumulative_seqlen_ptr[i + 1] = sum;

    // Find the max element while we traverse
    max_seqlen = std::max(max_seqlen, current_seq_len);
  }
  // Send to GPU, this is pretty light weight calc for normal batch size
  // but maybe this needs to be on gpu
  cumulative_seqlen = cumulative_seqlen.to(TensorOptions().device(at::kCUDA));
  return std::tuple<Tensor, int64_t, int64_t>{cumulative_seqlen, max_seqlen, sum};
}

/**
 * This function checks if a nested tensor is valid for
 * use with the flash-attention and efficient_attention kernels without
 * needing to call contiguous on the nested tensor input.
 * It checks that the storage offsets' adjacent_differences are a constant mutiple
 * of the previous tensor in the nested tensor and that the strides are monitonically decreasing.
 * This check is done after calling transpose on the nested tensor.
 *
 * @return A boolean indicating of contiguous needs to be called for input
 */
bool is_safe_to_get_storage_as_tensor(const NestedTensorImpl* tensor) {
  const int64_t *tensor_offsets_ptr = tensor->get_storage_offsets().data_ptr<int64_t>();
  const Tensor& tensor_sizes = tensor->get_nested_sizes();
  const Tensor& tensor_strides = tensor->get_nested_strides();

  const int64_t n_tensors = tensor_strides.size(0);
  const int64_t n_dims = tensor_strides.size(1);

  if (n_tensors <= 1) {
    return true;
  }

  int64_t* previous_tensor_stride = tensor_strides.data_ptr<int64_t>();
  // Check initially that they are in strictly descending order
  for (int i{1}; i < n_dims; i++) {
    if (previous_tensor_stride[i - 1] <= previous_tensor_stride[i]) {
      return false;
    }
  }
  // Check that each tensor i in the nested tensor has the same strides
  auto tensor_stride_0 = tensor_strides.stride(0);

  for (int i{1}; i < n_tensors; i++) {
    for (const int64_t j : c10::irange(n_dims)) {
      if (previous_tensor_stride[j] !=
          previous_tensor_stride[i * tensor_stride_0 + j]) {
        return false;
      }
    }
  }
  // Check the offsets are a constant multiple from the previous numels
  const int64_t* tensor_size_ptr = tensor_sizes.data_ptr<int64_t>();
  const int64_t* tensor_stride_ptr = tensor_strides.data_ptr<int64_t>();

  int64_t numel_0 = (tensor_size_ptr[0] * tensor_stride_ptr[0]);
  TORCH_INTERNAL_ASSERT(numel_0 > 0, "numels must be positive!");

  int64_t offset_constant = (tensor_offsets_ptr[1] - tensor_offsets_ptr[0]) / numel_0;
  for (int64_t i = 2; i < n_tensors; i++) {
    // TODO: When 0 seq_len nested tensors are allowed we need to guard against this
    int64_t previous_numel = tensor_size_ptr[(i - 1) * tensor_stride_0] * tensor_stride_ptr[(i - 1) * tensor_stride_0];
    TORCH_INTERNAL_ASSERT(previous_numel > 0, "numels must be positive!");
    int64_t current_offset_constant = (tensor_offsets_ptr[i] - tensor_offsets_ptr[i - 1]) / previous_numel;
    if (current_offset_constant != offset_constant) {
      return false;
    }
  }
  // Congrats you made it!
  return true;
}

/**
 * This function is a helper that takes nested query, key, and value
 * that require broadcasting on the batch or num_head dimensions
 * and will preprocess it in order to run with either
 * the flash-attention or efficient-attention kernels.
 * @return A tuple containing all the necessary data for running the fused kernels
 */
inline auto sdpa_nested_preprocessing_with_broadcast(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value) {
  // Query (Batch x Num_heads x {Q_seq_len}  x Dim_per_head)
  // Key   (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
  // Value (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
  const int64_t q_batch_size = query.size(0);
  const int64_t k_batch_size = key.size(0);
  const int64_t v_batch_size = value.size(0);

  const int64_t output_batch_size = std::max({q_batch_size, k_batch_size, v_batch_size});

  const int64_t q_num_heads = query.size(1);
  const int64_t k_num_heads = key.size(1);
  const int64_t v_num_heads = value.size(1);

  const int64_t output_num_heads = std::max({q_num_heads, k_num_heads, v_num_heads});

  const int64_t head_dim_qk = query.size(3);
  const int64_t head_dim_v = value.size(3);

  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);

  // Checks in sdp_utils ensure that if {*}_batch_size/{*}_num_heads != output_batch_size/num_heads
  // then they are 1
  bool q_batch_size_needs_broadcast = q_batch_size != output_batch_size;
  bool k_batch_size_needs_broadcast = k_batch_size != output_batch_size;
  bool v_batch_size_needs_broadcast = v_batch_size != output_batch_size;

  // If {*}_batch_size_needs_broadcast, then
  // (1) max_seqlen_batch_{*} is given by {*}_t.size(1)
  //     this is because needs_broadcast indicates that the batch_size is 1 and
  //     hence there is only 1 value for seq_len
  // (2) The cum_seq_lens are given by [0, {*}_t.size(1), 2 * {*}_t.size(1), ..., outut_batch_size * {*}_t.size(1)]
  // (3) Nnz_{*} is given by output_batch_size * {*}_t.size(1);

  int64_t max_seqlen_batch_q = 0, Nnz_q = 0;
  Tensor cumulative_sequence_length_q;
  if (q_batch_size_needs_broadcast || !q_t.is_nested()) {
    max_seqlen_batch_q = q_t.size(1);
    cumulative_sequence_length_q = at::arange(0,
                                              (output_batch_size + 1) * max_seqlen_batch_q,
                                              max_seqlen_batch_q,
                                              TensorOptions().device(at::kCUDA).dtype(at::kInt));
    Nnz_q = output_batch_size * max_seqlen_batch_q;
  } else {
    auto cumulative_and_max_q_and_nnz_q = cumulative_and_max_seq_len(q_t);
    cumulative_sequence_length_q = std::get<0>(cumulative_and_max_q_and_nnz_q);
    max_seqlen_batch_q = std::get<1>(cumulative_and_max_q_and_nnz_q);
    Nnz_q = std::get<2>(cumulative_and_max_q_and_nnz_q);
  }

  int64_t max_seqlen_batch_kv = 0, Nnz_kv = 0;
  Tensor cumulative_sequence_length_kv;
  if (k_batch_size_needs_broadcast && v_batch_size_needs_broadcast) {
    TORCH_CHECK(k_t.size(1) == v_t.size(1));
    max_seqlen_batch_kv = k_t.size(1);
    cumulative_sequence_length_kv = at::arange(0,
                                              (output_batch_size + 1) * max_seqlen_batch_kv,
                                              max_seqlen_batch_kv,
                                              TensorOptions().device(at::kCUDA).dtype(at::kInt));
    Nnz_kv = output_batch_size * max_seqlen_batch_kv;
  } else {
    auto cumulative_and_max_kv_and_nnz_kv = k_batch_size_needs_broadcast ?
                                            cumulative_and_max_seq_len(v_t) :
                                            cumulative_and_max_seq_len(k_t);
    cumulative_sequence_length_kv = std::get<0>(cumulative_and_max_kv_and_nnz_kv);
    max_seqlen_batch_kv = std::get<1>(cumulative_and_max_kv_and_nnz_kv);
    Nnz_kv = std::get<2>(cumulative_and_max_kv_and_nnz_kv);
  }

  bool q_num_heads_needs_broadcast = q_num_heads != output_num_heads;
  bool k_num_heads_needs_broadcast = k_num_heads != output_num_heads;
  bool v_num_heads_needs_broadcast = v_num_heads != output_num_heads;

  Tensor query_buffer_reshaped;
  Tensor key_buffer_reshaped;
  Tensor value_buffer_reshaped;

  const int64_t head_dim_stride = 1;

  // Generally the approach for q, k, v is to
  // (1) get the storage of the contiguous nested tensor
  // (2) view as shape {output_batch_size, {*}_t.size(1), output_num_heads, head_dim_{*}},
  //     and stride {0, nnz_{*}_stride, head_{*}_stride, head_dim_stride} where head_{*}_stride is 0
  //     if {*}_num_heads_needs_broadcast
  // (3) collapse the first two dims by reshaping to {Nnz_{*}, output_num_heads, head_dim_{*}}
  //     if {*}_t.size(1) (i.e. the seq_len is 1), the reshape should be a view and should not incur a copy
  // for q_t there is a dense path, so we can just use expand and reshape on the dense tensor
  // without getting the storage

  if (!q_t.is_nested()) {
    query_buffer_reshaped = q_t.expand({output_batch_size, q_t.size(1), output_num_heads, head_dim_qk});
    query_buffer_reshaped = query_buffer_reshaped.reshape({Nnz_q, output_num_heads, head_dim_qk});
  } else {
    const auto* query_impl = get_nested_tensor_impl(q_t);
    if (!q_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(query_impl)) {
      q_t = q_t.contiguous();
      query_impl = get_nested_tensor_impl(q_t);
    }
    Tensor q_storage_as_tensor =
      get_nested_tensor_impl(q_t)->get_unsafe_storage_as_tensor();
    auto query_stride_tensor = query_impl->get_nested_strides();

    const int64_t* q_strides = query_stride_tensor.data_ptr<int64_t>();
    const int64_t* q_offsets_ptr = query_impl->get_storage_offsets().data_ptr<int64_t>();
    const int64_t nnz_q_stride = q_strides[0];
    const int64_t head_q_stride = q_num_heads_needs_broadcast ? 0 : q_strides[1];

    if (q_batch_size_needs_broadcast) {
      query_buffer_reshaped = q_storage_as_tensor.as_strided(
          {output_batch_size, q_t.size(1), output_num_heads, head_dim_qk},
          {0, nnz_q_stride, head_q_stride, head_dim_stride},
          q_offsets_ptr[0]);
      // squash batch_size and seq_len dimensions.
      query_buffer_reshaped = query_buffer_reshaped.reshape({-1, output_num_heads, head_dim_qk});
    } else {
      query_buffer_reshaped = q_storage_as_tensor.as_strided(
          {Nnz_q, output_num_heads, head_dim_qk},
          {nnz_q_stride, head_q_stride, head_dim_stride},
          q_offsets_ptr[0]);
    }
  }

  const auto* key_impl = get_nested_tensor_impl(k_t);
  const auto* value_impl = get_nested_tensor_impl(v_t);

  // If the physical layout of the NestedTensor's storage
  // is not: batch, {seq_len}, num_heads, head_dim then we need
  // to call contiguous

  if (!k_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(key_impl)) {
    k_t = k_t.contiguous();
    key_impl = get_nested_tensor_impl(k_t);
  }
  if (!v_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(value_impl)) {
    v_t = v_t.contiguous();
    value_impl = get_nested_tensor_impl(v_t);
  }

  Tensor k_storage_as_tensor =
      get_nested_tensor_impl(k_t)->get_unsafe_storage_as_tensor();
  Tensor v_storage_as_tensor =
      get_nested_tensor_impl(v_t)->get_unsafe_storage_as_tensor();

  auto key_stride_tensor = key_impl->get_nested_strides();
  auto value_stride_tensor = value_impl->get_nested_strides();

  const int64_t* k_strides = key_stride_tensor.data_ptr<int64_t>();
  const int64_t* k_offsets_ptr = key_impl->get_storage_offsets().data_ptr<int64_t>();
  const int64_t nnz_k_stride = k_strides[0];
  const int64_t head_k_stride = k_num_heads_needs_broadcast ? 0 : k_strides[1];

  const int64_t* v_strides = value_stride_tensor.data_ptr<int64_t>();
  const int64_t* v_offsets_ptr = value_impl->get_storage_offsets().data_ptr<int64_t>();
  const int64_t nnz_v_stride = v_strides[0];
  const int64_t head_v_stride = v_num_heads_needs_broadcast ? 0 : v_strides[1];


  if (k_batch_size_needs_broadcast) {
    key_buffer_reshaped = k_storage_as_tensor.as_strided(
        {output_batch_size, k_t.size(1), output_num_heads, head_dim_qk},
        {0, nnz_k_stride, head_k_stride, head_dim_stride},
        k_offsets_ptr[0]);
    // squash batch_size and seq_len dimensions.
    key_buffer_reshaped = key_buffer_reshaped.reshape({-1, output_num_heads, head_dim_qk});
  } else {
    key_buffer_reshaped = k_storage_as_tensor.as_strided(
        {Nnz_kv, output_num_heads, head_dim_qk},
        {nnz_k_stride, head_k_stride, head_dim_stride},
        k_offsets_ptr[0]);
  }

  if (v_batch_size_needs_broadcast) {
    value_buffer_reshaped = v_storage_as_tensor.as_strided(
        {output_batch_size, v_t.size(1), output_num_heads, head_dim_v},
        {0, nnz_v_stride, head_v_stride, head_dim_stride},
        v_offsets_ptr[0]);
    // squash batch_size and seq_len dimensions.
    value_buffer_reshaped = value_buffer_reshaped.reshape({-1, output_num_heads, head_dim_v});
  } else {
    value_buffer_reshaped = v_storage_as_tensor.as_strided(
        {Nnz_kv, output_num_heads, head_dim_v},
        {nnz_v_stride, head_v_stride, head_dim_stride},
        v_offsets_ptr[0]);
  }

  Tensor output_shape;
  if (!q_batch_size_needs_broadcast) {
    output_shape = get_nested_sizes(q_t).clone();
    if (head_dim_v != head_dim_qk) {
      output_shape.select(1, -1).fill_(head_dim_v);
    }
    if (q_num_heads_needs_broadcast) {
      output_shape.select(1, 1).fill_(output_num_heads);
    }
  } else {
    output_shape = at::empty({output_batch_size, 3}, TensorOptions().dtype(kLong).device(kCPU));
    output_shape.select(1, 0).fill_(q_t.size(1));
    output_shape.select(1, 1).fill_(output_num_heads);
    output_shape.select(1, 2).fill_(head_dim_v);
  }

  return std::make_tuple(
      query_buffer_reshaped,
      key_buffer_reshaped,
      value_buffer_reshaped,
      cumulative_sequence_length_q,
      cumulative_sequence_length_kv,
      max_seqlen_batch_q,
      max_seqlen_batch_kv,
      output_shape);
}

/**
 * This function will take nested query, key, and value
 * and will preprocess it in order to run with either
 * the flash-attention or efficient-attention kernels.
 * @return A tuple containing all the necessary data for running the fused kernels
 */
inline auto sdpa_nested_preprocessing(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value) {
  // Query (Batch x Num_heads x {Q_seq_len}  x Dim_per_head)
  // Key   (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
  // Value (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
  const int64_t q_batch_size = query.size(0);
  const int64_t k_batch_size = key.size(0);
  const int64_t v_batch_size = value.size(0);

  const int64_t q_num_heads = query.size(1);
  const int64_t k_num_heads = key.size(1);
  const int64_t v_num_heads = value.size(1);

  if (!(q_batch_size == k_batch_size && q_batch_size == v_batch_size) ||
      !(q_num_heads == k_num_heads && k_num_heads == v_num_heads)) {
    return sdpa_nested_preprocessing_with_broadcast(query, key, value);
  }

  const int64_t num_heads = query.size(1);
  const int64_t head_dim_qk = query.size(3);
  const int64_t head_dim_v = value.size(3);

  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);

  auto cumulative_and_max_q_and_nnz_q = cumulative_and_max_seq_len(q_t);
  auto cumulative_and_max_kv_and_nnz_kv = cumulative_and_max_seq_len(k_t);

  // [TODO] K and V have to have the same Nnz, should probably torch_check
  // assume in order to not iterate over v

  Tensor cumulative_sequence_length_q =
      std::get<0>(cumulative_and_max_q_and_nnz_q);
  Tensor cumulative_sequence_length_kv =
      std::get<0>(cumulative_and_max_kv_and_nnz_kv);

  const int64_t max_seqlen_batch_q =
      std::get<1>(cumulative_and_max_q_and_nnz_q);
  const int64_t max_seqlen_batch_kv =
      std::get<1>(cumulative_and_max_kv_and_nnz_kv);

  const int64_t Nnz_q = std::get<2>(cumulative_and_max_q_and_nnz_q);
  const int64_t Nnz_kv = std::get<2>(cumulative_and_max_kv_and_nnz_kv);

  Tensor query_buffer_reshaped;
  Tensor key_buffer_reshaped;
  Tensor value_buffer_reshaped;

  const auto* query_impl = get_nested_tensor_impl(q_t);
  const auto* key_impl = get_nested_tensor_impl(k_t);
  const auto* value_impl = get_nested_tensor_impl(v_t);

  // If the physical layout of the NestedTensor's storage
  // is not: batch, {seq_len}, num_heads, head_dim then we need
  // to call contiguous
  if (!q_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(query_impl)) {
    q_t = q_t.contiguous();
    query_impl = get_nested_tensor_impl(q_t);
  }
  if (!k_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(key_impl)) {
    k_t = k_t.contiguous();
    key_impl = get_nested_tensor_impl(k_t);
  }
  if (!v_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(value_impl)) {
    v_t = v_t.contiguous();
    value_impl = get_nested_tensor_impl(v_t);
  }

  Tensor q_storage_as_tensor =
      get_nested_tensor_impl(q_t)->get_unsafe_storage_as_tensor();
  Tensor k_storage_as_tensor =
      get_nested_tensor_impl(k_t)->get_unsafe_storage_as_tensor();
  Tensor v_storage_as_tensor =
      get_nested_tensor_impl(v_t)->get_unsafe_storage_as_tensor();

  auto query_stride_tensor = query_impl->get_nested_strides();
  auto key_stride_tensor = key_impl->get_nested_strides();
  auto value_stride_tensor = value_impl->get_nested_strides();

  const int64_t head_dim_stride = 1;

  const int64_t* q_strides = query_stride_tensor.data_ptr<int64_t>();
  const int64_t* q_offsets_ptr = query_impl->get_storage_offsets().data_ptr<int64_t>();
  const int64_t nnz_q_stride = q_strides[0];
  const int64_t head_q_stride = q_strides[1];

  const int64_t* k_strides = key_stride_tensor.data_ptr<int64_t>();
  const int64_t* k_offsets_ptr = key_impl->get_storage_offsets().data_ptr<int64_t>();
  const int64_t nnz_k_stride = k_strides[0];
  const int64_t head_k_stride = k_strides[1];

  const int64_t* v_strides = value_stride_tensor.data_ptr<int64_t>();
  const int64_t* v_offsets_ptr = value_impl->get_storage_offsets().data_ptr<int64_t>();
  const int64_t nnz_v_stride = v_strides[0];
  const int64_t head_v_stride = v_strides[1];

  query_buffer_reshaped = q_storage_as_tensor.as_strided(
      {Nnz_q, num_heads, head_dim_qk},
      {nnz_q_stride, head_q_stride, head_dim_stride},
      q_offsets_ptr[0]);
  key_buffer_reshaped = k_storage_as_tensor.as_strided(
      {Nnz_kv, num_heads, head_dim_qk},
      {nnz_k_stride, head_k_stride, head_dim_stride},
      k_offsets_ptr[0]);
  value_buffer_reshaped = v_storage_as_tensor.as_strided(
      {Nnz_kv, num_heads, head_dim_v},
      {nnz_v_stride, head_v_stride, head_dim_stride},
      v_offsets_ptr[0]);

  auto output_shape = get_nested_sizes(q_t).clone();
  if (head_dim_v != head_dim_qk) {
    output_shape.select(1, -1).fill_(head_dim_v);
  }

  return std::make_tuple(
      query_buffer_reshaped,
      key_buffer_reshaped,
      value_buffer_reshaped,
      cumulative_sequence_length_q,
      cumulative_sequence_length_kv,
      max_seqlen_batch_q,
      max_seqlen_batch_kv,
      output_shape);
}

} // namespace

std::tuple<
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    int64_t,
    int64_t,
    Tensor,
    Tensor,
    Tensor>
_scaled_dot_product_flash_attention_nestedtensor_cuda(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    c10::optional<double> scale) {
  Tensor query_buffer_reshaped, key_buffer_reshaped, value_buffer_reshaped,
      cumulative_sequence_length_q, cumulative_sequence_length_kv, output_shape;
  int64_t max_seqlen_batch_q{0}, max_seqlen_batch_kv{0};
  std::tie(
      query_buffer_reshaped,
      key_buffer_reshaped,
      value_buffer_reshaped,
      cumulative_sequence_length_q,
      cumulative_sequence_length_kv,
      max_seqlen_batch_q,
      max_seqlen_batch_kv,
      output_shape) = sdpa_nested_preprocessing(query, key, value);

  Tensor attention, log_sumexp, debug_attn_mask, philox_seed, philox_offset;
  std::tie(attention, log_sumexp, philox_seed, philox_offset, debug_attn_mask) =
      at::_flash_attention_forward(
          query_buffer_reshaped,
          key_buffer_reshaped,
          value_buffer_reshaped,
          cumulative_sequence_length_q,
          cumulative_sequence_length_kv,
          max_seqlen_batch_q,
          max_seqlen_batch_kv,
          dropout_p,
          is_causal,
          return_debug_mask,
          scale);
  // Reshape output to convert nnz to batch_size and seq_len
  attention = wrap_buffer(attention.view(-1), output_shape).transpose(1, 2);
  return std::make_tuple(
      attention,
      log_sumexp,
      cumulative_sequence_length_q,
      cumulative_sequence_length_kv,
      max_seqlen_batch_q,
      max_seqlen_batch_kv,
      philox_seed,
      philox_offset,
      debug_attn_mask);
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
_scaled_dot_product_efficient_attention_nestedtensor_cuda(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<at::Tensor>&  attn_bias,
    bool compute_log_sumexp,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale) {
  Tensor query_buffer_reshaped, key_buffer_reshaped, value_buffer_reshaped,
      cumulative_sequence_length_q, cumulative_sequence_length_kv, output_shape;
  int64_t max_seqlen_batch_q{0};
  std::tie(
      query_buffer_reshaped,
      key_buffer_reshaped,
      value_buffer_reshaped,
      cumulative_sequence_length_q,
      cumulative_sequence_length_kv,
      max_seqlen_batch_q,
      std::ignore,
      output_shape) = sdpa_nested_preprocessing(query, key, value);

  sdp::CustomMaskType custom_mask_type = is_causal
      ? sdp::CustomMaskType::CausalFromTopLeft
      : sdp::CustomMaskType::NoCustomMask;

  // See Note [Seed and Offset] for description of seed and offset
  auto [attention, log_sumexp, seed, offset] = at::_efficient_attention_forward(
      query_buffer_reshaped.unsqueeze(0),
      key_buffer_reshaped.unsqueeze(0),
      value_buffer_reshaped.unsqueeze(0),
      c10::nullopt,
      cumulative_sequence_length_q,
      cumulative_sequence_length_kv,
      max_seqlen_batch_q,
      dropout_p,
      static_cast<int64_t>(custom_mask_type),
      compute_log_sumexp,
      scale);

  // Reshape output to convert nnz to batch_size and seq_len
  attention = wrap_buffer(attention.view(-1), output_shape).transpose(1, 2);
  return std::make_tuple(std::move(attention), std::move(log_sumexp), std::move(seed), std::move(offset));
}

} // namespace native
} // namespace at
