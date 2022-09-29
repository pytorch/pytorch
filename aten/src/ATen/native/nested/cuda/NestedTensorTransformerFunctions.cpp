#include <type_traits>
#include <c10/util/Exception.h>

#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_nested_from_padded.h>
#endif

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
    if (padded.dtype() != kFloat && padded.dtype() != kHalf) {
      TORCH_WARN_ONCE(
          "nested_from_padded CUDA kernels only support fp32/fp16; falling "
          "back to slower generic kernel");
      return at::native::nested_from_padded_generic(padded, sizes, do_transform_0213);
    }
    TORCH_CHECK(
        (padded.dim() == 4 && do_transform_0213) ||
            (padded.dim() == 3 && !do_transform_0213),
        "padded tensor size error");
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
  int64_t* offsets_ptr = offsets.data_ptr<int64_t>();
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
      Tensor nt_sizes = nt_input->get_nested_size_tensor();
      Tensor sizes_dim1 = at::native::narrow(nt_sizes, 1, 0, 1);
      Tensor sizes_dim2 = at::native::narrow(nt_sizes, 1, 1, 1);
      Tensor result = at::detail::make_tensor<NestedTensorImpl>(
          nt_input->get_buffer(), sizes_dim1 * sizes_dim2[0]);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.dim() == 2);
      result =
          NestedTensor_to_padded_tensor_cuda(result, padding, output_size);
      return result.reshape({result.sizes()[0], -1, *nt_input->opt_size(2)});
    }

    Tensor nt_sizes = nt_input->get_nested_size_tensor();
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

std::tuple<Tensor, Tensor> _scaled_dot_product_attention_forward_nested(
        const Tensor& query_, const Tensor& key, const Tensor& value,
        const c10::optional<Tensor>& attn_mask_, double dropout_p, bool need_attn_weights, bool is_causal) {

    // Determine which efficient kernel to use
    sdp::sdp_params kernel_params{query_, key, value, attn_mask_.has_value(), dropout_p, need_attn_weights, is_causal};
    auto backend = select_sdp_backend(kernel_params);
    switch(backend){
      case sdp::SDPBackend::flash_attention:
          return flash_attention_helper_nested_unpacked(query_, key, value, dropout_p, need_attn_weights, is_causal);
      case sdp::SDPBackend::math:
        return at::_scaled_dot_product_attention_math(query_, key, value, attn_mask_, dropout_p, need_attn_weights, is_causal);
      default:
        TORCH_CHECK(false, "Unsupported backend for scaled_dot_product_attention");
        return std::make_tuple(Tensor(), Tensor());
    }
}


std::tuple<Tensor, int64_t> cumulative_and_max_seq_len(Tensor qkv) {
  TORCH_CHECK(
      qkv.is_nested(),
      "QKV must be nested for flash cumulative_seq_len calculation.")
  auto* nt_impl = get_nested_tensor_impl(qkv);
  const auto& sizes = nt_impl->get_nested_size_tensor();
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
    auto current_seq_len = sizes_ptr[i * size_tensor_stride];
    sum += current_seq_len;
    cumulative_seqlen_ptr[i + 1] = sum;

    // Find the max element while we traverse
    max_seqlen = std::max(max_seqlen, current_seq_len);
  }
  // Send to GPU, this is pretty light weight calc for normal batch size
  // but maybe this needs to be on gpu
  cumulative_seqlen = cumulative_seqlen.to(TensorOptions().device(at::kCUDA));
  return std::tuple<Tensor, int64_t>{cumulative_seqlen, max_seqlen};
}
std::tuple<Tensor, Tensor> flash_attention_helper_nested_unpacked(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool need_atten_weights,
    bool is_causal) {
  // Query (Batch x Num_heads x {Q_seq_len}  x Dim_per_head)
  // Key   (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
  // Value (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
  const int64_t num_heads = query.size(1);
  const int64_t head_dim = query.size(3);

  // Query -> Query (Batch x {Q_seq_len}  x Num_heads x Dim_per_head)
  // Key   -> Key   (Batch x {KV_seq_len} x Num_heads x Dim_per_head)
  // Value -> Value (Batch x {KV_seq_len} x Num_heads x Dim_per_head)
  Tensor q_t = query.transpose(1, 2).contiguous();
  Tensor k_t = key.transpose(1, 2).contiguous();
  Tensor v_t = value.transpose(1, 2).contiguous();

  // K and V have to have the same Nnz, should probably torch_check
  // assume in order to not iterate over v

  auto cumulative_and_max_q = cumulative_and_max_seq_len(q_t);
  auto cumulative_and_max_k = cumulative_and_max_seq_len(k_t);

  Tensor cumulative_sequence_length_q = std::get<0>(cumulative_and_max_q);
  Tensor cumulative_sequence_length_k = std::get<0>(cumulative_and_max_k);

  const int64_t max_seqlen_batch_q = std::get<1>(cumulative_and_max_q);
  const int64_t max_seqlen_batch_k = std::get<1>(cumulative_and_max_k);

  const int64_t Nnz_q  = cumulative_sequence_length_q[-1].item<int64_t>();
  const int64_t Nnz_kv = cumulative_sequence_length_k[-1].item<int64_t>();

  auto query_buffer_reshaped =
      get_buffer(q_t).view({Nnz_q, num_heads, head_dim});
  auto key_buffer_reshaped =
      get_buffer(k_t).view({Nnz_kv, num_heads, head_dim});
  auto value_buffer_reshaped =
      get_buffer(v_t).view({Nnz_kv, num_heads, head_dim});

  std::tuple<Tensor,Tensor> attention_and_weights =
  at::_flash_scaled_dot_product_attention(
      query_buffer_reshaped,
      key_buffer_reshaped,
      value_buffer_reshaped,
      cumulative_sequence_length_q,
      cumulative_sequence_length_k,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      dropout_p,
      need_atten_weights,
      is_causal);
  // Reshape output to convert nnz to batch_size and seq_len
  Tensor attention = std::get<0>(attention_and_weights);
  attention = wrap_buffer(attention.view(-1), get_nested_size_tensor(q_t).clone()).transpose(1,2);
  return std::tie(attention, std::get<1>(attention_and_weights));
}

std::tuple<Tensor, Tensor> flash_attention_helper(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool need_atten_weights,
    bool is_causal) {
  //  Query is of size (batch_size x ragged_seq_len x (3 or 1) x n_heads x
  //  head_did
  int64_t head_dim{query.size(-1)};
  int64_t num_heads{query.size(-2)};

  auto cumulative_and_max_q = cumulative_and_max_seq_len(query);
  Tensor cumulative_sequence_length_q = std::get<0>(cumulative_and_max_q);
  int64_t max_seqlen_batch_q = std::get<1>(cumulative_and_max_q);

  if (key.is_same(value) || query.is_same(key) || query.is_same(value)) {
    int64_t Nnz_q{cumulative_sequence_length_q[-1].item<int64_t>()};

    // For the packed case we need to set the output size for dim 2 to 1
    auto atten_size = get_nested_size_tensor(query).clone();
    atten_size.index({at::indexing::Slice(), 1}) = 1;

    auto qkv_buffer_reshaped =
        get_buffer(query).view({Nnz_q, 3, num_heads, head_dim}).transpose(0, 1).contiguous();

    auto q = qkv_buffer_reshaped[0];
    auto k = qkv_buffer_reshaped[1];
    auto v = qkv_buffer_reshaped[2];

    TORCH_CHECK(q.is_contiguous());
    TORCH_CHECK(k.is_contiguous());
    TORCH_CHECK(v.is_contiguous());

    // If we are passing in query, key, value all the same tensors then we have
    // packed them into one tensor and need to slice for flash attention
    std::tuple<Tensor,Tensor> attention_and_weights = at::_flash_scaled_dot_product_attention(
        q,
        k,
        v,
        cumulative_sequence_length_q,
        cumulative_sequence_length_q,
        max_seqlen_batch_q,
        max_seqlen_batch_q,
        dropout_p,
        need_atten_weights,
        is_causal);
    // Output of flash_attention is a regular tensor lets wrap it back up to
    // form a nested tensor
    std::get<0>(attention_and_weights) = wrap_buffer(std::get<0>(attention_and_weights).view(-1), atten_size);
    return attention_and_weights;
  }

  // Query, Key, and Value are not all the same tensor and therefore need to
  // calculate K meta data

  // The nested tensors will be of shape {Batch_size x ragged_seq_len x
  // num_heads * head_dim }
  auto cumulative_and_max_k = cumulative_and_max_seq_len(key);
  Tensor cumulative_sequence_length_k = std::get<0>(cumulative_and_max_k);
  int64_t max_seqlen_batch_k = std::get<1>(cumulative_and_max_k);

  // K and V have to have the same Nnz, should probably torch_check before now
  // assume in order to not iterate over v
  int64_t Nnz_q{cumulative_sequence_length_q[-1].item<int64_t>()};
  int64_t Nnz_kv{cumulative_sequence_length_k[-1].item<int64_t>()};

  auto query_buffer_reshaped =
      get_buffer(query).view({Nnz_q, num_heads, head_dim}).contiguous();
  auto key_buffer_reshaped =
      get_buffer(key).view({Nnz_kv, num_heads, head_dim}).contiguous();
  auto value_buffer_reshaped =
      get_buffer(value).view({Nnz_kv, num_heads, head_dim}).contiguous();

  std::tuple<Tensor,Tensor> attention_and_weights = at::_flash_scaled_dot_product_attention(
      query_buffer_reshaped,
      key_buffer_reshaped,
      value_buffer_reshaped,
      cumulative_sequence_length_q,
      cumulative_sequence_length_k,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      dropout_p,
      need_atten_weights,
      is_causal);
  // Output of flash_attention is a regular tensor lets wrap it back up to
  // form a nested tensor, the size of which should match the query tensor
  std::get<0>(attention_and_weights) = wrap_buffer(std::get<0>(attention_and_weights).view(-1), get_nested_size_tensor(query).clone());
  return attention_and_weights;
}

} // namespace native
} // namespace at
