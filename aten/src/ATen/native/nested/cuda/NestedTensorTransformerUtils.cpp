#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorTransformerUtils.h>
#include <tuple>

namespace at::native::preprocessing {

namespace {

/**
 * This builds up the cumulative sequence length for a batch of sequences.
 * This is not very dry, but in the backward pass we already have cumulative_seq_len
 * on device. And all we need on CPU to launch the kernel is NNz. We could refactor the
 * the below function but it adds more complexity than I think is needed.
 */
int64_t get_nnz(const Tensor& nestedtensor) {
  auto* nt_impl = get_nested_tensor_impl(nestedtensor);
  const auto& sizes = nt_impl->get_nested_sizes();
  auto size_tensor_stride = sizes.stride(0);
  const int64_t batch_size = nestedtensor.size(0);
  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  int64_t cumulative_sequence_length = 0;
  for (const auto i : c10::irange(batch_size)) {
    // Calculate the cumulative sum of the sequence lengths
    int64_t current_seq_len = sizes_ptr[(i * size_tensor_stride)];
    cumulative_sequence_length += current_seq_len;
  }
  return cumulative_sequence_length;
}

  /**
   * This function is used to calculate two pieces of metadata that are needed
   * for use with flash-attention and efficient_attention kernels. They are the
   * cumulative sequence_length over a batch of sequences and the maximum
   * sequence length.
   *
   * @return A tuple of cumulative sequence lengths and the maximum sequence
   * length, and the last element in the cumulative_sequence_lengths
   */
  std::tuple<Tensor, int64_t, int64_t> cumulative_and_max_seq_len_nnz(const Tensor& qkv) {
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

    int64_t sum = 0;
    int64_t max_seqlen = -1;
    cumulative_seqlen_ptr[0] = static_cast<int32_t>(sum);
    for (const auto i : c10::irange(batch_size)) {
      // Calculate the cumulative sum of the sequence lengths
      auto current_seq_len = sizes_ptr[(i * size_tensor_stride)];
      sum += current_seq_len;
      cumulative_seqlen_ptr[i + 1] = static_cast<int32_t>(sum);

      // Find the max element while we traverse
      max_seqlen = std::max(max_seqlen, current_seq_len);
    }
    // Send to GPU, this is pretty light weight calc for normal batch size
    // but maybe this needs to be on gpu
    cumulative_seqlen = cumulative_seqlen.to(TensorOptions().device(at::kCUDA));
    return std::tuple<Tensor, int64_t, int64_t>{
        cumulative_seqlen, max_seqlen, sum};
  }

  /**
   * This function checks if a nested tensor is valid for
   * use with the flash-attention and efficient_attention kernels without
   * needing to call contiguous on the nested tensor input.
   * It checks that the storage offsets' adjacent_differences are a constant
   * multiple of the previous tensor in the nested tensor and that the strides
   * are monotonically decreasing. This check is done after calling transpose on
   * the nested tensor. Resulting in a Nt of shape [bsz, {seq_len}, num_heads, dim]
   *
   * @return A boolean indicating of contiguous needs to be called for input
   */
  bool is_safe_to_get_storage_as_tensor(const NestedTensorImpl* tensor) {
    const int64_t* tensor_offsets_ptr =
        tensor->get_storage_offsets().data_ptr<int64_t>();
    const Tensor& tensor_sizes = tensor->get_nested_sizes();
    const Tensor& tensor_strides = tensor->get_nested_strides();

    const int64_t n_tensors = tensor_strides.size(0);
    constexpr int n_dims = 3;
    // This is safe since head_dim is assured to be consistent
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const int64_t num_heads = tensor -> opt_size(2).value();
    const int64_t tensor_stride_0 = tensor_strides.stride(0);

    if (n_tensors <= 1) {
      return true;
    }

    int64_t* previous_tensor_stride = tensor_strides.data_ptr<int64_t>();

    // Check initially that the first tensor's strides
    // are in strictly descending order
    // NOTE: If num_heads is equal to 1 then we skip stride[0]
    // Why you may ask? This is because we if n_heads == 1 then
    // then as long as the last stride == 1 it does not matter
    // what the strides are for the other dimensions.
    //
    if (num_heads == 1) {
      if (previous_tensor_stride[0] <= previous_tensor_stride[2]) {
        // This would mean that the last stride is greater than the seq_len
        // stride
        return false;
      }
    } else {
      for (int i{1}; i < n_dims; i++) {
        if (previous_tensor_stride[i - 1] <= previous_tensor_stride[i]) {
          return false;
        }
      }
      // Check that each tensor i in the nested tensor has the same strides
      for (int64_t i{1}; i < n_tensors; i++) {
        for (const int64_t j : c10::irange(n_dims)) {
          if (previous_tensor_stride[j] !=
              previous_tensor_stride[i * tensor_stride_0 + j]) {
            return false;
          }
        }
      }
    }

    // Check the offsets are a constant multiple from the previous numels
    const int64_t* tensor_size_ptr = tensor_sizes.const_data_ptr<int64_t>();
    const int64_t* tensor_stride_ptr = tensor_strides.const_data_ptr<int64_t>();

    int64_t numel_0 = (tensor_size_ptr[0] * tensor_stride_ptr[0]);
    TORCH_INTERNAL_ASSERT(numel_0 > 0, "numels must be positive!");

    int64_t offset_constant =
        (tensor_offsets_ptr[1] - tensor_offsets_ptr[0]) / numel_0;
    for (int64_t i = 2; i < n_tensors; i++) {
      // TODO: When 0 seq_len nested tensors are allowed we need to guard
      // against this
      int64_t previous_numel = tensor_size_ptr[(i - 1) * tensor_stride_0] *
          tensor_stride_ptr[(i - 1) * tensor_stride_0];
      TORCH_INTERNAL_ASSERT(previous_numel > 0, "numels must be positive!");
      int64_t current_offset_constant =
          (tensor_offsets_ptr[i] - tensor_offsets_ptr[i - 1]) / previous_numel;
      if (current_offset_constant != offset_constant) {
        return false;
      }
    }
    // Congrats you made it!
    return true;
  }

  /**
   * Process an individual NestedTensor to reshape and view as a DenseTensor
   * Generally the approach for q, k, v is to
   * (1) get the storage of the contiguous nested tensor
   * (2) view as shape {output_batch_size, {*}_t.size(1), output_num_heads,
   * head_dim_{*}}, and stride {0, nnz_{*}_stride, head_{*}_stride,
   * head_dim_stride} where head_{*}_stride is 0 if
   * {*}_num_heads_needs_broadcast (3) collapse the first two dims by reshaping
   * to {Nnz_{*}, output_num_heads, head_dim_{*}} if {*}_t.size(1) (i.e. the
   * seq_len is 1), the reshape should be a view and should not incur a copy
   *  dense tensor without getting the storage
   */
  at::Tensor view_as_dense(
      const at::Tensor& input_nestedtensor,
      const int64_t Nnz,
      const int64_t num_heads,
      const int64_t head_dim,
      const bool batch_needs_broadcast = false,
      const bool num_heads_needs_broadcast = false) {
    const auto* tensor_impl = get_nested_tensor_impl(input_nestedtensor);
    Tensor storage_as_tensor = tensor_impl->get_unsafe_storage_as_tensor();

    constexpr int64_t head_dim_stride = 1;
    const int64_t* nt_strides =
        tensor_impl->get_nested_strides().data_ptr<int64_t>();
    const int64_t* nt_offsets_ptr =
        tensor_impl->get_storage_offsets().data_ptr<int64_t>();

    const int64_t nnz_stride = nt_strides[0];
    const int64_t head_stride = num_heads_needs_broadcast ? 0 : nt_strides[1];

    if (batch_needs_broadcast) {
      Tensor input_buffer_reshaped = storage_as_tensor.as_strided(
          {Nnz, input_nestedtensor.size(1), num_heads, head_dim},
          {0, nnz_stride, head_stride, head_dim_stride},
          nt_offsets_ptr[0]);
      return input_buffer_reshaped.reshape({-1, num_heads, head_dim});
    }
    return storage_as_tensor.as_strided(
        {Nnz, num_heads, head_dim},
        {nnz_stride, head_stride, head_dim_stride},
        nt_offsets_ptr[0]);
  }

  /**
   * This function is a helper that takes nested query, key, and value
   * that require broadcasting on the batch or num_head dimensions
   * and will preprocess it in order to run with either
   * the flash-attention or efficient-attention kernels.
   * @return A tuple containing all the necessary data for running the fused
   * kernels
   */
  auto sdpa_nested_preprocessing_with_broadcast(
      const Tensor& query, const Tensor& key, const Tensor& value) {
    // Query (Batch x Num_heads x {Q_seq_len}  x Dim_per_head)
    // Key   (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
    // Value (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
    const int64_t q_batch_size = query.size(0);
    const int64_t k_batch_size = key.size(0);
    const int64_t v_batch_size = value.size(0);

    const int64_t output_batch_size =
        std::max({q_batch_size, k_batch_size, v_batch_size});

    const int64_t q_num_heads = query.size(1);
    const int64_t k_num_heads = key.size(1);
    const int64_t v_num_heads = value.size(1);

    const int64_t output_num_heads =
        std::max({q_num_heads, k_num_heads, v_num_heads});

    const int64_t head_dim_qk = query.size(3);
    const int64_t head_dim_v = value.size(3);

    Tensor q_t = query.transpose(1, 2);
    Tensor k_t = key.transpose(1, 2);
    Tensor v_t = value.transpose(1, 2);

    // Checks in sdp_utils ensure that if {*}_batch_size/{*}_num_heads !=
    // output_batch_size/num_heads then they are 1
    bool q_batch_size_needs_broadcast = q_batch_size != output_batch_size;
    bool k_batch_size_needs_broadcast = k_batch_size != output_batch_size;
    bool v_batch_size_needs_broadcast = v_batch_size != output_batch_size;

    // If {*}_batch_size_needs_broadcast, then
    // (1) max_seqlen_batch_{*} is given by {*}_t.size(1)
    //     this is because needs_broadcast indicates that the batch_size is 1
    //     and hence there is only 1 value for seq_len
    // (2) The cum_seq_lens are given by [0, {*}_t.size(1), 2 * {*}_t.size(1),
    // ..., outut_batch_size * {*}_t.size(1)] (3) Nnz_{*} is given by
    // output_batch_size * {*}_t.size(1);

    int64_t max_seqlen_batch_q = 0, Nnz_q = 0;
    Tensor cumulative_sequence_length_q;
    if (q_batch_size_needs_broadcast || !q_t.is_nested()) {
      max_seqlen_batch_q = q_t.size(1);
      cumulative_sequence_length_q = at::arange(
          0,
          (output_batch_size + 1) * max_seqlen_batch_q,
          max_seqlen_batch_q,
          TensorOptions().device(at::kCUDA).dtype(at::kInt));
      Nnz_q = output_batch_size * max_seqlen_batch_q;
    } else {
      std::tie(cumulative_sequence_length_q, max_seqlen_batch_q, Nnz_q) = cumulative_and_max_seq_len_nnz(q_t);
    }

    int64_t max_seqlen_batch_kv = 0, Nnz_kv = 0;
    Tensor cumulative_sequence_length_kv;
    if (k_batch_size_needs_broadcast && v_batch_size_needs_broadcast) {
      TORCH_CHECK(k_t.size(1) == v_t.size(1));
      max_seqlen_batch_kv = k_t.size(1);
      cumulative_sequence_length_kv = at::arange(
          0,
          (output_batch_size + 1) * max_seqlen_batch_kv,
          max_seqlen_batch_kv,
          TensorOptions().device(at::kCUDA).dtype(at::kInt));
      Nnz_kv = output_batch_size * max_seqlen_batch_kv;
    } else {
      std::tie(cumulative_sequence_length_kv, max_seqlen_batch_kv, Nnz_kv) =
      k_batch_size_needs_broadcast
          ? cumulative_and_max_seq_len_nnz(v_t)
          : cumulative_and_max_seq_len_nnz(k_t);
    }

    bool q_num_heads_needs_broadcast = q_num_heads != output_num_heads;
    bool k_num_heads_needs_broadcast = k_num_heads != output_num_heads;
    bool v_num_heads_needs_broadcast = v_num_heads != output_num_heads;

    Tensor query_buffer_reshaped;
    Tensor key_buffer_reshaped;
    Tensor value_buffer_reshaped;

    if (!q_t.is_nested()) {
      query_buffer_reshaped = q_t.expand(
          {output_batch_size, q_t.size(1), output_num_heads, head_dim_qk});
      query_buffer_reshaped =
          query_buffer_reshaped.reshape({Nnz_q, output_num_heads, head_dim_qk});
    } else {
      const auto* query_impl = get_nested_tensor_impl(q_t);
      if (!q_t.is_contiguous() &&
          !is_safe_to_get_storage_as_tensor(query_impl)) {
        q_t = q_t.contiguous();
      }
      // If we are broadcasting then Nnz_q will be the output_batch_size since
      // seq_len is 1
      const int64_t effective_batch_size_q =
          q_batch_size_needs_broadcast ? output_batch_size : Nnz_q;
      query_buffer_reshaped = view_as_dense(
          q_t,
          effective_batch_size_q,
          output_num_heads,
          head_dim_qk,
          q_batch_size_needs_broadcast,
          q_num_heads_needs_broadcast);
    }

    const auto* key_impl = get_nested_tensor_impl(k_t);
    const auto* value_impl = get_nested_tensor_impl(v_t);

    // If the physical layout of the NestedTensor's storage
    // is not: batch, {seq_len}, num_heads, head_dim then we need
    // to call contiguous

    if (!k_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(key_impl)) {
      k_t = k_t.contiguous();
    }
    if (!v_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(value_impl)) {
      v_t = v_t.contiguous();
    }
    const int64_t effective_batch_size_k =
        k_batch_size_needs_broadcast ? output_batch_size : Nnz_kv;
    key_buffer_reshaped = view_as_dense(
        k_t,
        effective_batch_size_k,
        output_num_heads,
        head_dim_qk,
        k_batch_size_needs_broadcast,
        k_num_heads_needs_broadcast);

    const int64_t effective_batch_size_v =
        v_batch_size_needs_broadcast ? output_batch_size : Nnz_kv;
    value_buffer_reshaped = view_as_dense(
        v_t,
        effective_batch_size_v,
        output_num_heads,
        head_dim_v,
        v_batch_size_needs_broadcast,
        v_num_heads_needs_broadcast);

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
      output_shape = at::empty(
          {output_batch_size, 3}, TensorOptions().dtype(kLong).device(kCPU));
      output_shape.select(1, 0).fill_(q_t.size(1));
      output_shape.select(1, 1).fill_(output_num_heads);
      output_shape.select(1, 2).fill_(head_dim_v);
    }

    return std::make_tuple(
        std::move(query_buffer_reshaped),
        std::move(key_buffer_reshaped),
        std::move(value_buffer_reshaped),
        std::move(cumulative_sequence_length_q),
        std::move(cumulative_sequence_length_kv),
        max_seqlen_batch_q,
        max_seqlen_batch_kv,
        std::move(output_shape));
  }

} // namespace

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, int64_t, int64_t, Tensor>
sdpa_nested_preprocessing(
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

  auto [cumulative_sequence_length_q, max_seqlen_batch_q, Nnz_q] = cumulative_and_max_seq_len_nnz(q_t);
  auto [cumulative_sequence_length_kv, max_seqlen_batch_kv, Nnz_kv]= cumulative_and_max_seq_len_nnz(k_t);

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
  }
  if (!k_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(key_impl)) {
    k_t = k_t.contiguous();
  }
  if (!v_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(value_impl)) {
    v_t = v_t.contiguous();
  }

  query_buffer_reshaped = view_as_dense(q_t, Nnz_q, num_heads, head_dim_qk);
  key_buffer_reshaped = view_as_dense(k_t, Nnz_kv, num_heads, head_dim_qk);
  value_buffer_reshaped = view_as_dense(v_t, Nnz_kv, num_heads, head_dim_v);

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

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
sdpa_nested_preprocessing_backward(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const Tensor& cumulative_sequence_length_q,
    const Tensor& cumulative_sequence_length_kv,
    const int64_t max_seqlen_batch_q,
    const int64_t max_seqlen_batch_kv) {
  const int64_t q_batch_size = query.size(0);
  const int64_t k_batch_size = key.size(0);

  const int64_t v_batch_size = value.size(0);

  const int64_t q_num_heads = query.size(1);
  const int64_t k_num_heads = key.size(1);
  const int64_t v_num_heads = value.size(1);

  if (!(q_batch_size == k_batch_size && q_batch_size == v_batch_size) ||
      !(q_num_heads == k_num_heads && k_num_heads == v_num_heads)) {
        TORCH_CHECK(false, "Broadcasted NestedTensor inputs is currently not supported for backwards.");
  }

  const int64_t num_heads = query.size(1);
  const int64_t head_dim_qk = query.size(3);
  const int64_t head_dim_v = value.size(3);

  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);
  Tensor grad_out_t = grad_out_.transpose(1, 2);
  Tensor out_t = out.transpose(1, 2);

  const int64_t Nnz_q = get_nnz(q_t);
  const int64_t Nnz_kv = get_nnz(k_t);

  Tensor query_buffer_reshaped;
  Tensor key_buffer_reshaped;
  Tensor value_buffer_reshaped;
  Tensor grad_out_buffer_reshaped;
  Tensor output_buffer_reshaped;

  const auto* query_impl = get_nested_tensor_impl(q_t);
  const auto* key_impl = get_nested_tensor_impl(k_t);
  const auto* value_impl = get_nested_tensor_impl(v_t);
  const auto* grad_out_impl = get_nested_tensor_impl(grad_out_t);
  const auto* out_impl = get_nested_tensor_impl(out_t);

  // If the physical layout of the NestedTensor's storage
  // is not: batch, {seq_len}, num_heads, head_dim then we need
  // to call contiguous
  if (!q_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(query_impl)) {
    q_t = q_t.contiguous();
  }
  if (!k_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(key_impl)) {
    k_t = k_t.contiguous();
  }
  if (!v_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(value_impl)) {
    v_t = v_t.contiguous();
  }
  if (!grad_out_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(grad_out_impl)) {
    grad_out_t = grad_out_t.contiguous();
  }
  if (!out_t.is_contiguous() && !is_safe_to_get_storage_as_tensor(out_impl)) {
    out_t = out_t.contiguous();
  }

  query_buffer_reshaped = view_as_dense(q_t, Nnz_q, num_heads, head_dim_qk);
  key_buffer_reshaped = view_as_dense(k_t, Nnz_kv, num_heads, head_dim_qk);
  value_buffer_reshaped = view_as_dense(v_t, Nnz_kv, num_heads, head_dim_v);

 grad_out_buffer_reshaped =
      view_as_dense(grad_out_t, Nnz_q, num_heads, head_dim_v);
  output_buffer_reshaped = view_as_dense(out_t, Nnz_q, num_heads, head_dim_v);

  auto output_shape = get_nested_sizes(q_t).clone();
  if (head_dim_v != head_dim_qk) {
    output_shape.select(1, -1).fill_(head_dim_v);
  }

  return std::make_tuple(
      grad_out_buffer_reshaped,
      query_buffer_reshaped,
      key_buffer_reshaped,
      value_buffer_reshaped,
      output_buffer_reshaped);
}

} // namespace at::native::preprocessing
