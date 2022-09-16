#include <ATen/native/nested/NestedTensorTransformerFunctions.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#include <c10/util/string_view.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

namespace at {
namespace native {
namespace {

inline void check_nested_tensor_matrix_constraints(
    const Tensor& nested_tensor,
    const Tensor& dense_matrix,
    c10::string_view caller) {
  auto* nt_input = get_nested_tensor_impl(nested_tensor);
  TORCH_INTERNAL_ASSERT(nt_input != nullptr);
  TORCH_CHECK(
      !dense_matrix.is_nested(),
      caller,
      " does not support nested weight when input is a nested tensor.")
  // TODO: support noncontiguous case
  // error out for now
  TORCH_CHECK(
      nested_tensor_impl_is_contiguous(nt_input),
      "for now linear only supports contiguous nested tensor");
  TORCH_CHECK(
      nested_tensor.dim() == 3 && dense_matrix.dim() == 2,
      caller,
      " requires nested_tensor.dim == 3 and dense_matrix.dim == 2."
      " Nested tensor dim: ",
      nested_tensor.dim(),
      ". Dense tensor dim: ",
      dense_matrix.dim());
  const auto last_dim = get_consistent_last_dim_of_nested_tensor(*nt_input);
  // We check check the second dimension for linear because it transposes before matrix multiply
  int64_t dim_constraint = (caller == "Linear") ? 1 : 0;
  auto dense_size = dense_matrix.size(dim_constraint);
  TORCH_CHECK(
      last_dim == dense_size,
      "Shape mismatch for NestedTensor ",
      caller,
      ": Expected input's (a nested tensor) 'last_dim' to equal 'weight.size(",
      dim_constraint,
      "),",
      " but got: last_dim = ",
      last_dim,
      ", and weight.size(",
      dim_constraint,
      ") = ",
      dense_size);
}
} // namespace

Tensor nested_linear(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt) {
  check_nested_tensor_matrix_constraints(input, weight, c10::string_view{"Linear"});
  auto* nt_input = get_nested_tensor_impl(input);
  const Tensor& input_buffer = nt_input->get_buffer();
  Tensor result_buffer =
      at::linear(input_buffer.reshape({-1, weight.size(1)}), weight, bias_opt);
  result_buffer = result_buffer.reshape({-1});
  int64_t weight_size_1 = weight.size(0);
  Tensor new_sizes = nt_input->get_nested_size_tensor().clone();
  // Now the last entry in every row of new_sizes should be weight_size_1.
  new_sizes.index_put_({at::indexing::Slice(), -1}, weight_size_1);
  return wrap_buffer(result_buffer, new_sizes);
}

Tensor NestedTensor_matmul(const Tensor& self, const Tensor& other) {
  check_nested_tensor_matrix_constraints(self, other, c10::string_view{"Matmul"});
  auto* nt_self = get_nested_tensor_impl_or_null(self);
  const Tensor& self_buffer = nt_self->get_buffer();
  Tensor result_buffer =
      at::mm(self_buffer.reshape({-1, other.sizes()[0]}), other);
  result_buffer = result_buffer.reshape({-1});
  int64_t other_size_1 = other.sizes()[1];
  Tensor new_sizes = nt_self->get_nested_size_tensor().clone();
  // Now the last entry in every row of new_sizes should be other_size_1.
  new_sizes.index_put_({at::indexing::Slice(), -1}, other_size_1);
  return wrap_buffer(result_buffer, new_sizes);
}

Tensor NestedTensor_times_Tensor_plus_Tensor_addmm(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const c10::Scalar& beta,
    const c10::Scalar& alpha,
    c10::optional<bool> use_gelu) {
  // Interesting case: alpha * NT * T + beta * T
  const auto* nt_mat1 = get_nested_tensor_impl_or_null(mat1);
  TORCH_INTERNAL_ASSERT(nt_mat1 != nullptr);
  TORCH_INTERNAL_ASSERT(!mat2.is_nested());
  TORCH_INTERNAL_ASSERT(!self.is_nested());
  TORCH_INTERNAL_ASSERT(nested_tensor_impl_is_contiguous(nt_mat1));
  TORCH_INTERNAL_ASSERT(mat1.dim() == 3 && mat2.dim() == 2);
  TORCH_INTERNAL_ASSERT(
      get_consistent_last_dim_of_nested_tensor(*nt_mat1) == mat2.sizes()[0]);
  const Tensor& mat1_buffer = nt_mat1->get_buffer();
  Tensor result_buffer = !use_gelu.has_value()
      ? at::addmm(
            self, mat1_buffer.reshape({-1, mat2.sizes()[0]}), mat2, beta, alpha)
      : at::_addmm_activation(
            self,
            mat1_buffer.reshape({-1, mat2.sizes()[0]}),
            mat2,
            beta,
            alpha,
            *use_gelu);
  result_buffer = result_buffer.reshape({-1});
  int64_t other_size_1 = mat2.sizes()[1];
  Tensor new_sizes = nt_mat1->get_nested_size_tensor().clone();
  new_sizes.index_put_({at::indexing::Slice(), -1}, other_size_1);
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(result_buffer), std::move(new_sizes));
}

Tensor NestedTensor_add_NestedTensor_in_place(
    const Tensor& self,
    const Tensor& other) {
  TORCH_INTERNAL_ASSERT(self.is_nested() && other.is_nested());
  const auto& nt_self = *get_nested_tensor_impl(self);
  const auto& nt_other = *get_nested_tensor_impl(other);

  const auto& self_sizes = nt_self.get_nested_size_tensor();
  const auto& other_sizes = nt_other.get_nested_size_tensor();

  TORCH_CHECK(at::equal(self_sizes, other_sizes));
  TORCH_INTERNAL_ASSERT(
      nested_tensor_impl_is_contiguous(&nt_self) &&
      nested_tensor_impl_is_contiguous(&nt_other));
  nt_self.get_buffer().view({-1}).add_(nt_other.get_buffer().view({-1}));
  return self;
}

Tensor NestedTensor_softmax_dropout(const Tensor& self, const Tensor& query) {
  const auto* query_nt = get_nested_tensor_impl_or_null(query);
  TORCH_INTERNAL_ASSERT(query_nt != nullptr);
  TORCH_INTERNAL_ASSERT(nested_tensor_impl_is_contiguous(query_nt));

  const Tensor& sizes = query_nt->get_nested_size_tensor();
  const auto num_tensors = sizes.sizes()[0];

  auto output = at::empty_like(self,{}, at::MemoryFormat::Contiguous);
  TORCH_INTERNAL_ASSERT(output.is_contiguous());

  const auto max_seq_len = self.sizes()[2];

  for (int64_t i = 0; i < num_tensors; i++) {
    auto seq_len = sizes.index({i, 0}).item<int64_t>();
    auto subseq = self.index(
        {i,
         indexing::Slice(),
         indexing::Slice(0, seq_len),
         indexing::Slice(0, seq_len)});
    auto subscores = at::softmax(subseq, subseq.dim() - 1);
    output.index_put_(
        {i,
         indexing::Slice(),
         indexing::Slice(0, seq_len),
         indexing::Slice(0, seq_len)},
        subscores);
    output.index_put_(
        {i,
         indexing::Slice(),
         indexing::Slice(0, seq_len),
         indexing::Slice(seq_len, max_seq_len)},
        0);
    output.index_put_(
        {i,
         indexing::Slice(),
         indexing::Slice(seq_len, max_seq_len),
         indexing::Slice(0, max_seq_len)},
        0);
  }
  return output;
}

Tensor NestedTensor_softmax_dropout_cuda(const Tensor& self, const Tensor& query) {
  c10::optional<Tensor> attn_mask;

  attn_mask = NestedTensor_to_mask(query, 2, self.size(2));
  attn_mask = attn_mask->to(query.device(), /*non-blocking=*/true);
  return _masked_softmax(self, *attn_mask, self.dim() - 1, /*mask type */ 1 );  // NestedTensor_to_mask produces a BxT mask
}

Tensor NestedTensor_batch_offsets_from_size_tensor(
    const Tensor& sizes,
    int64_t extra_elements) {
  int64_t* const sizes_ptr = sizes.data_ptr<int64_t>();
  Tensor offsets = at::empty({1 + sizes.size(0) + extra_elements}, at::kInt);
  int32_t* const offsets_ptr = offsets.data_ptr<int32_t>();
  offsets_ptr[0] = 0;
  const auto sizes_size_1 = sizes.size(1);
  const auto sizes_size_0 = sizes.size(0);
  for (const auto i : c10::irange(sizes_size_0)) {
    int64_t prod = 1;
    for (const auto j : c10::irange(sizes_size_1)) {
      prod *= sizes_ptr[i * sizes_size_1 + j];
    }
    offsets_ptr[i + 1] = offsets_ptr[i] + prod;
  }
  return offsets;
}


Tensor NestedTensor_to_mask(const Tensor& nt, c10::optional<int64_t> mask_dim, c10::optional<int64_t> mask_dim_length) {
  auto* nt_impl = get_nested_tensor_impl(nt);
  TORCH_CHECK(
      !mask_dim || *mask_dim < nt.dim(),
      "Requested mask dimension ",
      *mask_dim,
      " is bigger than dimension ",
      nt.dim(),
      " of given NestedTensor.");

  // TODO: port optimization for 1x1 tensors from
  // pytorch/nestedtensor's version.

  TORCH_CHECK(
      mask_dim && *mask_dim == 2 && nt.dim() == 3,
      "Only the special case of mask_dim == 2 on a 3-D NestedTensor is supported right now.")
  const auto& sizes = nt_impl->get_nested_size_tensor();
  // Shape: # of tensors in our NestedTensor by max size along first dim
  // TODO: calculate this without allocating a std::vector.
  const auto result_size_1 = mask_dim_length ? *mask_dim_length : NestedTensor_get_max_size(*nt_impl)[0];
  auto result = at::ones({sizes.sizes()[0], result_size_1}, at::kBool);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(sizes.dim() == 2);
  auto* result_data = result.data_ptr<bool>();
  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  const auto sizes_size_1 = sizes.sizes()[1];
  for (const auto ii : c10::irange(sizes.sizes()[0])) {
    auto length = sizes_ptr[ii * sizes_size_1];
    for (const auto jj : c10::irange(length)) {
      result_data[ii * result_size_1 + jj] = false;
    }
  }
  return result;
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

Tensor flash_attention_helper(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool causal) {
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

    auto i0 = qkv_buffer_reshaped[0];
    auto i1 = qkv_buffer_reshaped[1];
    auto i2 = qkv_buffer_reshaped[2];

    TORCH_CHECK(i0.is_contiguous());
    TORCH_CHECK(i1.is_contiguous());
    TORCH_CHECK(i2.is_contiguous());

    // If we are passing in query, key, value all the same tensors then we have
    // packed them into one tensor and need to slice for flash attention
    Tensor atten_buffer = at::_flash_scaled_dot_product_attention(
        i0,
        i1,
        i2,
        cumulative_sequence_length_q,
        cumulative_sequence_length_q,
        max_seqlen_batch_q,
        max_seqlen_batch_q,
        dropout_p,
        causal);
    // Output of flash_attention is a regular tensor lets wrap it back up to
    // form a nested tensor
    return wrap_buffer(atten_buffer.view(-1), atten_size);
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
      get_buffer(query).view({Nnz_q, num_heads, head_dim});
  auto key_buffer_reshaped =
      get_buffer(key).view({Nnz_kv, num_heads, head_dim});
  auto value_buffer_reshaped =
      get_buffer(value).view({Nnz_kv, num_heads, head_dim});

  Tensor atten_buffer = at::_flash_scaled_dot_product_attention(
      query_buffer_reshaped,
      key_buffer_reshaped,
      value_buffer_reshaped,
      cumulative_sequence_length_q,
      cumulative_sequence_length_k,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      dropout_p,
      causal);
  // Output of flash_attention is a regular tensor lets wrap it back up to
  // form a nested tensor, the size of which should match the query tensor
  return wrap_buffer(atten_buffer.view(-1), get_nested_size_tensor(query));
}

Tensor flash_attention_helper_dense(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool causal) {
  TORCH_INTERNAL_ASSERT(
      !query.is_nested() && !key.is_nested() && !value.is_nested());
  //  Query is of size (batch_size x dense_seq_len x 3 x n_heads
  //  head_dim)
  const auto batch_size = query.size(0);
  auto max_seqlen_batch_q = query.size(1);
  int64_t head_dim{query.size(-1)};
  int64_t num_heads{query.size(-2)};

  auto cumulative_sequence_length_q = at::arange(
      0,
      (batch_size + 1) * max_seqlen_batch_q,
      max_seqlen_batch_q,
      TensorOptions().device(at::kCUDA).dtype(at::kInt));
  int64_t Nnz_q{batch_size * max_seqlen_batch_q};

  if (key.is_same(value) || query.is_same(key) || query.is_same(value)) {
    // In the dense case flash attention expects an input that is
    // (b*s) x num_heads x head_dim
    auto query_reshaped = query.reshape({Nnz_q, 3, num_heads, head_dim});
    // If we are passing in query, key, value all the same tensors than we have
    // packed them into one tensor and need to slice for flash attention

    Tensor atten_buffer = at::_flash_scaled_dot_product_attention(
        query_reshaped.index({at::indexing::Slice(), 0}),
        query_reshaped.index({at::indexing::Slice(), 1}),
        query_reshaped.index({at::indexing::Slice(), 2}),
        cumulative_sequence_length_q,
        cumulative_sequence_length_q,
        max_seqlen_batch_q,
        max_seqlen_batch_q,
        dropout_p,
        causal);
    // Reshape output to convert nnz to batch_size and seq_len
    return atten_buffer.reshape(
        {batch_size, max_seqlen_batch_q, num_heads, head_dim});
  }

  // Query, Key, and Value are not all the same tensor and therefore need to
  // calculate K meta data
  auto max_seqlen_batch_k = key.size(1);
  auto cumulative_sequence_length_k = at::arange(
      0,
      (batch_size + 1) * max_seqlen_batch_k,
      max_seqlen_batch_k,
      TensorOptions().device(at::kCUDA).dtype(at::kInt));

  // K and V have to have the same Nnz, should probably torch_check before
  // assume for now in order to not iterate over v
  int64_t Nnz_kv{batch_size * max_seqlen_batch_k};

  // Calculate head dim
  TORCH_INTERNAL_ASSERT(query.size(-1) == key.size(-1));
  TORCH_INTERNAL_ASSERT(query.size(-1) == value.size(-1));

  auto query_reshaped = query.reshape({Nnz_q, num_heads, head_dim});
  auto key_reshaped = key.reshape({Nnz_kv, num_heads, head_dim});
  auto value_reshaped = value.reshape({Nnz_kv, num_heads, head_dim});

  Tensor atten_buffer = at::_flash_scaled_dot_product_attention(
      query_reshaped,
      key_reshaped,
      value_reshaped,
      cumulative_sequence_length_q,
      cumulative_sequence_length_k,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      dropout_p,
      causal);
  // Reshape output to convert nnz to batch_size and seq_len
  return atten_buffer.reshape(
      {batch_size, max_seqlen_batch_q, num_heads, head_dim});
}

} // namespace native
} // namespace at
