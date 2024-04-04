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
#include <ATen/ops/_nested_from_padded_native.h>
#include <ATen/ops/narrow_native.h>
#endif

#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
#include <ATen/native/nested/NestedTensorTransformerUtils.h>
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
  TORCH_CHECK(t.numel() > 0, "to_padded_tensor: at least one constituent tensor should have non-zero numel")
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

std::tuple<
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    c10::SymInt,
    c10::SymInt,
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
      output_shape) = preprocessing::sdpa_nested_preprocessing(query, key, value);

  auto
      [attention,
       logsumexp,
       philox_seed,
       philox_offset,
       debug_attn_mask] =
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
      logsumexp,
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
  int64_t max_seqlen_batch_k{0};
  std::tie(
      query_buffer_reshaped,
      key_buffer_reshaped,
      value_buffer_reshaped,
      cumulative_sequence_length_q,
      cumulative_sequence_length_kv,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      output_shape) = preprocessing::sdpa_nested_preprocessing(query, key, value);

  sdp::CustomMaskType custom_mask_type = is_causal
      ? sdp::CustomMaskType::CausalFromTopLeft
      : sdp::CustomMaskType::NoCustomMask;

  // See Note [Seed and Offset] for description of seed and offset
  // Although max_seqlen_q, and max_seqlen_batch_kv is returned we drop these values.
  auto [attention, log_sumexp, seed, offset, max_seqlen_q, max_seqlen_batch_kv] = at::_efficient_attention_forward(
      query_buffer_reshaped.unsqueeze(0),
      key_buffer_reshaped.unsqueeze(0),
      value_buffer_reshaped.unsqueeze(0),
      c10::nullopt,
      cumulative_sequence_length_q,
      cumulative_sequence_length_kv,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      dropout_p,
      static_cast<int64_t>(custom_mask_type),
      compute_log_sumexp,
      scale);

  // Reshape output to convert nnz to batch_size and seq_len
  attention = wrap_buffer(attention.view(-1), output_shape).transpose(1, 2);
  return std::make_tuple(std::move(attention), std::move(log_sumexp), std::move(seed), std::move(offset));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _scaled_dot_product_flash_attention_backward_nested(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const Tensor& cumulative_sequence_length_q,
    const Tensor& cumulative_sequence_length_k,
    const int64_t max_seqlen_batch_q,
    const int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    c10::optional<double> scale){
  if (!grad_out_.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{});
  }
  Tensor grad_out_buffer_reshaped, query_buffer_reshaped, key_buffer_reshaped,
      value_buffer_reshaped, output_buffer_reshaped;
  std::tie(
      grad_out_buffer_reshaped,
      query_buffer_reshaped,
      key_buffer_reshaped,
      value_buffer_reshaped,
      output_buffer_reshaped) =
      preprocessing::sdpa_nested_preprocessing_backward(
          grad_out_,
          query,
          key,
          value,
          out,
          cumulative_sequence_length_q,
          cumulative_sequence_length_k,
          max_seqlen_batch_q,
          max_seqlen_batch_k);

  Tensor grad_q, grad_k, grad_v;
  std::tie(grad_q, grad_k, grad_v) = at::_flash_attention_backward(
    grad_out_buffer_reshaped,
    query_buffer_reshaped,
    key_buffer_reshaped,
    value_buffer_reshaped,
    output_buffer_reshaped,
    logsumexp,
    cumulative_sequence_length_q,
    cumulative_sequence_length_k,
    max_seqlen_batch_q,
    max_seqlen_batch_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    scale);

  grad_q = wrap_buffer(grad_q.view(-1), query.transpose(1,2)._nested_tensor_size()).transpose(1,2);
  grad_k = wrap_buffer(grad_k.view(-1), key.transpose(1,2)._nested_tensor_size()).transpose(1,2);
  grad_v = wrap_buffer(grad_v.view(-1), value.transpose(1,2)._nested_tensor_size()).transpose(1,2);

  return std::make_tuple(grad_q, grad_k, grad_v);
}

} // namespace native
} // namespace at
