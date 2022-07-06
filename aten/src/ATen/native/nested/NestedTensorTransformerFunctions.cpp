#include <ATen/native/nested/NestedTensorTransformerFunctions.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorMath.h>
#include <c10/util/string_view.h>
#include <c10/util/Exception.h>
#include <c10/core/TensorOptions.h>

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
  TORCH_CHECK(nested_tensor_impl_is_contiguous(nt_input));
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

std::tuple<Tensor, Tensor, Tensor> nested_linear_backward(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    std::array<bool, 3> output_mask) {
  if (!grad_output.defined()) {
    return std::tuple<Tensor, Tensor, Tensor>{Tensor(), Tensor(), Tensor()};
  }
  Tensor grad_input, grad_weight, grad_bias;
  auto* nt_grad_output = get_nested_tensor_impl(grad_output);
  auto* nt_input = get_nested_tensor_impl(input);
  TORCH_INTERNAL_ASSERT(nt_grad_output != nullptr);
  TORCH_INTERNAL_ASSERT(nt_input != nullptr);
  TORCH_CHECK(nested_tensor_impl_is_contiguous(nt_grad_output));
  auto grad_ouput_buffer = nt_grad_output->get_buffer();
  auto input_buffer = nt_input->get_buffer();

  auto reshaped_grad = grad_ouput_buffer.reshape({-1, weight.size(0)});

  if (output_mask[0]) {
    auto grad_input_buffer = at::mm(reshaped_grad, weight).view({-1});
    auto grad_input_nt_size = nt_input->get_nested_size_tensor().clone();
    grad_input = wrap_buffer(grad_input_buffer, grad_input_nt_size);
  }
  if (output_mask[1]) {
    grad_weight =
        at::mm(reshaped_grad.t(), input_buffer.reshape({-1, weight.size(1)}));
  }
  if (output_mask[2]) {
    grad_bias = reshaped_grad.sum(0);
  }
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
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

void NestedTensor_softmax_dropout(const Tensor& query, Tensor& attn_scores) {
  const auto* query_nt = get_nested_tensor_impl_or_null(query);
  TORCH_INTERNAL_ASSERT(query_nt != nullptr);
  TORCH_INTERNAL_ASSERT(nested_tensor_impl_is_contiguous(query_nt));

  const Tensor& sizes = query_nt->get_nested_size_tensor();
  const auto num_tensors = sizes.sizes()[0];
  const auto max_seq_len = attn_scores.sizes()[2];

  for (int64_t i = 0; i < num_tensors; i++) {
    auto seq_len = sizes.index({i, 0}).item<int64_t>();
    auto subseq = attn_scores.index(
        {i,
         indexing::Slice(),
         indexing::Slice(0, seq_len),
         indexing::Slice(0, seq_len)});
    auto subscores = at::softmax(subseq, subseq.dim() - 1);
    attn_scores.index_put_(
        {i,
         indexing::Slice(),
         indexing::Slice(0, seq_len),
         indexing::Slice(0, seq_len)},
        subscores);
    attn_scores.index_put_(
        {i,
         indexing::Slice(),
         indexing::Slice(0, seq_len),
         indexing::Slice(seq_len, max_seq_len)},
        0);
    attn_scores.index_put_(
        {i,
         indexing::Slice(),
         indexing::Slice(seq_len, max_seq_len),
         indexing::Slice(0, max_seq_len)},
        0);
  }
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

Tensor NestedTensor_to_mask(
    const Tensor& nt,
    c10::optional<int64_t> mask_dim,
    c10::optional<int64_t> mask_dim_length) {
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
  const auto result_size_1 = mask_dim_length
      ? *mask_dim_length
      : NestedTensor_get_max_size(*nt_impl)[0];
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
  auto cumulative_seqlen = at::zeros({batch_size + 1}, TensorOptions().dtype(at::kInt));

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

Tensor flash_scaled_dot_product_self_attention(
    const Tensor& qkv,
    const Tensor& cumulative_sequence_length,
    const int64_t max_seqlen_batch,
    const int64_t num_heads,
    double dropout_p,
    bool causal) {
  TORCH_CHECK(false, "This function needs to be overriden from python");
  return at::Tensor();
}

Tensor NestedTensor_scaled_dot_product_self_attention(
    const Tensor& qkv,
    int64_t num_heads,
    double dropout_p,
    bool causal) {
  // We assume that the input has already been projected
  TORCH_CHECK(qkv.is_nested(), "QKV tensor must be nested");
  // qkv nested_size -> batch_size x ragged_seq_len x (3 * num_heads * head_dim)
  const auto embed_dim = qkv.size(-1);
  TORCH_CHECK(
      embed_dim % (num_heads * 3) == 0,
      "Expected embedding dim to be divisible by 3*num_heads");
  if (true) {
    //  Hot path for flash attention
    auto cumulative_and_max = cumulative_and_max_seq_len(qkv);
    Tensor cumulative_tensor = std::get<0>(cumulative_and_max);
    int64_t max_seq_len = std::get<1>(cumulative_and_max);

    int64_t head_dim = embed_dim / (num_heads * 3);
    int64_t Nnz = cumulative_tensor[-1].item<int64_t>();

    auto qkv_buffer_reshaped =
        get_buffer(qkv).view({Nnz, 3, num_heads, head_dim});
    Tensor atten_bufer = at::_flash_scaled_dot_product_self_attention(
        qkv, cumulative_tensor, max_seq_len, num_heads, dropout_p, causal);
    Tensor atten_nt = wrap_buffer(atten_bufer, get_nested_size_tensor(qkv));
  }

  auto qkv_transposed = NestedTensor_transpose(qkv);
  auto q_dot_k_scaled = at::bmm(qkv, qkv_transposed).softmax(-1);
  at::dropout_(q_dot_k_scaled, dropout_p, false);
  auto atten = at::bmm(q_dot_k_scaled, qkv);
  return atten;
}

} // namespace native
} // namespace at
