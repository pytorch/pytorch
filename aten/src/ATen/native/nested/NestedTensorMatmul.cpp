#include <ATen/native/nested/NestedTensorMath.h>
#include <ATen/native/nested/NestedTensorUtils.h>

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
#include <ATen/core/grad_mode.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#include <tuple>

namespace at {
namespace native {

Tensor bmm_nested(const Tensor& self, const Tensor& mat2) {
  TORCH_CHECK(self.dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(mat2.dim() == 3, "batch2 must be a 3D tensor");

  int64_t ntensors = self.is_nested() ? get_nested_tensor_impl(self)->size(0) : self.size(0);
  int64_t ntensors2 = mat2.is_nested() ? get_nested_tensor_impl(mat2)->size(0) : mat2.size(0);

  TORCH_CHECK(ntensors == ntensors2,
      "Expected size for the 1st dimension of batch2 tensor to be: ", ntensors,
      " but got: ", ntensors2, ".");

  const Tensor& self_buffer = self.is_nested() ? get_nested_tensor_impl(self)->get_unsafe_storage_as_tensor() : self;
  const Tensor& mat2_buffer = mat2.is_nested() ? get_nested_tensor_impl(mat2)->get_unsafe_storage_as_tensor() : mat2;


  // create a contiguous output
  int64_t out_numel = 0;
  const Tensor& self_sizemat = self.is_nested() ?
      get_nested_tensor_impl(self)->get_nested_sizes() : get_nested_tensor_impl(mat2)->get_nested_sizes();

  Tensor out_sizemat = self_sizemat.new_empty(self_sizemat.sizes());
  int64_t* out_sizemat_ptr = out_sizemat.data_ptr<int64_t>();
  for (int64_t i = 0; i < ntensors; i++) {
    const IntArrayRef& self_shape = get_size_for_index(self, i);
    const IntArrayRef& mat2_shape = get_size_for_index(mat2, i);
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
  Tensor out_buffer = self.is_nested() ? self_buffer.new_empty(out_numel) : mat2_buffer.new_empty(out_numel);
  Tensor output = wrap_buffer(out_buffer, out_sizemat);
  // call tensor mm
  // TODO: `padding nested tensor -> bmm -> remove padding` may be more efficient
  //       until we have specialized nested tensor bmm kernel
  //       useful resource: `aten/src/ATen/native/cpu/LinearAlgebra.cpp/bmm_out_or_baddbmm_`
  //                        `aten/src/ATen/native/cuda/Blas.cpp/baddbmm_out_cuda_impl`
  std::vector<Tensor> output_unbind = output.unbind();
  for (int64_t i = 0; i < ntensors; i++) {
    at::mm_out(output_unbind[i],
              self_buffer.as_strided(get_size_for_index(self, i), get_stride_for_index(self, i), get_offset_for_index(self, i)),
              mat2_buffer.as_strided(get_size_for_index(mat2, i), get_stride_for_index(mat2, i), get_offset_for_index(mat2, i)));
  }
  return output;
}



static Tensor matmul_with_bmm_nested(const Tensor& self, const Tensor& mat2) {
  // Tensor self = self_.contiguous();
  // Tensor mat2 = mat2_.contiguous();
  // self [N, n_heads, *, head_dim]
  // mat2 [N, n_heads, head_dim, *]
  const auto self_ptr = get_nested_tensor_impl(self);
  const auto mat2_ptr = get_nested_tensor_impl(mat2);
  // metadata for self
  std::vector<IntArrayRef> self_sizes = NestedTensor_get_sizes(self_ptr);
  std::vector<IntArrayRef> self_strides = NestedTensor_get_strides(self_ptr);
  int64_t* self_offsets_ptr =
      self_ptr->get_storage_offsets().data_ptr<int64_t>();
  auto opt = self_ptr->get_nested_sizes().options();

  // metadata for mat2
  std::vector<IntArrayRef> mat2_sizes = NestedTensor_get_sizes(mat2_ptr);
  std::vector<IntArrayRef> mat2_strides = NestedTensor_get_strides(mat2_ptr);
  int64_t* mat2_offsets_ptr =
      mat2_ptr->get_storage_offsets().data_ptr<int64_t>();
  auto opt2 = mat2_ptr->get_nested_sizes().options();

  int64_t N = self_sizes.size();
  int64_t n_heads = self_sizes[0][0];

  // viewed metadata for self
  auto self_new_sizes = at::empty({N * n_heads, 2}, opt);
  int64_t* self_new_sizes_ptr = self_new_sizes.mutable_data_ptr<int64_t>();

  auto self_new_strides = at::empty({N * n_heads, 2}, opt);
  int64_t* self_new_strides_ptr = self_new_strides.mutable_data_ptr<int64_t>();
  auto self_new_offsets = at::empty({N * n_heads}, opt);
  int64_t* self_new_offsets_ptr = self_new_offsets.mutable_data_ptr<int64_t>();

  // viewed metadata for mat2
  auto mat2_new_sizes = at::empty({N * n_heads, 2}, opt2);
  int64_t* mat2_new_sizes_ptr = mat2_new_sizes.mutable_data_ptr<int64_t>();

  auto mat2_new_strides = at::empty({N * n_heads, 2}, opt2);
  int64_t* mat2_new_strides_ptr = mat2_new_strides.mutable_data_ptr<int64_t>();
  auto mat2_new_offsets = at::empty({N * n_heads}, opt);
  int64_t* mat2_new_offsets_ptr = mat2_new_offsets.mutable_data_ptr<int64_t>();

  for (int64_t i = 0; i < N; i++) {
    const IntArrayRef& self_size_i = self_sizes[i];
    const IntArrayRef& self_stride_i = self_strides[i];
    int64_t self_offset = self_offsets_ptr[i];

    const IntArrayRef& mat2_size_i = mat2_sizes[i];
    const IntArrayRef& mat2_stride_i = mat2_strides[i];
    int64_t mat2_offset = mat2_offsets_ptr[i];
    for (int64_t j = 0; j < n_heads; j++) {
      auto idx = (i * n_heads + j) * 2;
      self_new_sizes_ptr[idx] = self_size_i[1];
      self_new_sizes_ptr[idx + 1] = self_size_i[2];
      self_new_strides_ptr[idx] = self_stride_i[1];
      self_new_strides_ptr[idx + 1] = self_stride_i[2];
      auto offset_idx = i * n_heads + j;
      self_new_offsets_ptr[offset_idx] = self_offset;
      self_offset += self_stride_i[0];

      mat2_new_sizes_ptr[idx] = mat2_size_i[1];
      mat2_new_sizes_ptr[idx + 1] = mat2_size_i[2];
      mat2_new_strides_ptr[idx] = mat2_stride_i[1];
      mat2_new_strides_ptr[idx + 1] = mat2_stride_i[2];
      mat2_new_offsets_ptr[offset_idx] = mat2_offset;
      mat2_offset += mat2_stride_i[0];
    }
  }

  // view self as [N * n_heads, *, head_dim] (collapse first 2 dims)
  auto viewed_self = create_nested_view_tensor(
      self, self_new_sizes, self_new_strides, self_new_offsets);

  // view mat2 as [N * n_heads, head_dim, *] (collapse first 2_dims)
  auto viewed_mat2 = create_nested_view_tensor(
      mat2, mat2_new_sizes, mat2_new_strides, mat2_new_offsets);

  // output [N * n_heads, *, *]
  auto bmm_output = at::bmm(viewed_self, viewed_mat2);

  // generate metadata for viewing output as [N, n_heads, *, *]
  // output of bmm should be contiguous so stride calculations should hold
  auto out_new_sizes = at::empty({N, 3}, opt);
  auto out_new_strides = at::empty({N, 3}, opt);
  auto out_new_offsets = at::empty({N}, opt);
  int64_t* out_new_offsets_ptr = out_new_offsets.mutable_data_ptr<int64_t>();

  int64_t* out_new_sizes_ptr = out_new_sizes.data_ptr<int64_t>();
  int64_t* out_new_strides_ptr = out_new_strides.data_ptr<int64_t>();

  int64_t out_offset = 0;
  for (int64_t i = 0; i < N; i++) {
    out_new_offsets_ptr[i] = out_offset;
    const IntArrayRef& self_size_i = self_sizes[i];
    const IntArrayRef& mat2_size_i = mat2_sizes[i];
    auto idx = i * 3;
    out_new_sizes_ptr[idx] = n_heads;
    out_new_sizes_ptr[idx + 1] = self_size_i[1];
    out_new_sizes_ptr[idx + 2] = mat2_size_i[2];
    out_new_strides_ptr[idx] = self_size_i[1] * mat2_size_i[2];
    out_new_strides_ptr[idx + 1] = mat2_size_i[2];
    out_new_strides_ptr[idx + 2] = 1;
    out_offset += n_heads * (self_size_i[1] * mat2_size_i[2]);
  }

  auto viewed_out = create_nested_view_tensor(
      bmm_output, out_new_sizes, out_new_strides, out_new_offsets);

  return viewed_out;
}

// nt: NT of shape (B, *, C, D)
// other: dense tensor of shape (D, E)
// output: NT of shape (B, *, C, E)
static Tensor matmul_nested_with_broadcasted_dense(
    const Tensor& nt,
    const Tensor& other) {
  // View nt buffer as 3D jagged for matmul
  auto* nt_impl = get_nested_tensor_impl(nt);
  auto jagged = nt_impl->get_buffer().view({-1, nt.size(2), nt.size(3)});
  auto new_buffer = at::matmul(jagged, other);

  // Wrap result into nested tensor
  const auto E = other.size(-1);
  const auto component_dim = nt.dim() - 1;
  auto new_sizes = nt_impl->get_nested_sizes().clone();
  auto new_sizes_ptr = new_sizes.data_ptr<int64_t>();
  for (const auto i : c10::irange(nt.size(0))) {
    new_sizes_ptr[i * component_dim + 2] = E;
  }
  return at::detail::make_tensor<NestedTensorImpl>(
      new_buffer.view(-1), new_sizes);
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
  // special case of NT (B, *, C, D) with broadcasted dense (D, E)
  if (self.is_nested() && self.is_contiguous() && !mat2.is_nested() &&
      self.dim() == 4 && mat2.dim() == 2 &&
      get_nested_tensor_impl(self)->opt_size(2).has_value() &&
      get_nested_tensor_impl(self)->opt_size(3).has_value() &&
      self.size(3) == mat2.size(0)) {
    return matmul_nested_with_broadcasted_dense(self, mat2);
  }
  if (self.is_nested() && !mat2.is_nested()) {
    AT_ERROR(
        "Expected both to be nested, but got a nested self and non-nested other");
  } else if (!self.is_nested() && mat2.is_nested()) {
    AT_ERROR(
        "Expected both to be nested, but got a non-nested self and nested other");
  }
  // to_padded_tensor only supports contiguous inputs
  auto self_contig = self.contiguous();
  auto mat2_contig = mat2.contiguous();
  // dispatcher should have guaranteed that at least one is nested
  const auto self_ptr = get_nested_tensor_impl(self_contig);
  const auto mat2_ptr = get_nested_tensor_impl(mat2_contig);
  int64_t self_dim = self_ptr->dim(), mat2_dim = mat2_ptr->dim();
  TORCH_CHECK(
      self_dim >= 3,
      "matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: ",
      self_dim);
  TORCH_CHECK(
      mat2_dim >= 3,
      "matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 2nd input has rank: ",
      mat2_dim);
  TORCH_CHECK(
      self_dim == mat2_dim, "matmul: both inputs must have the same rank");
  int64_t ntensors = self_ptr->size(0), ntensors2 = mat2_ptr->size(0);
  TORCH_CHECK(
      ntensors == ntensors2,
      "matmul: Expected size for the 1st dimension of 2nd input tensor to be: ",
      ntensors,
      " but got: ",
      ntensors2,
      ".");
  // Ensure batch dimensions have the same sizes (no broadcasting).
  const auto& self_sizes = self_ptr->get_nested_sizes();
  const auto& mat2_sizes = mat2_ptr->get_nested_sizes();
  const auto& self_batch_sizes = self_sizes.narrow(1, 0, self_dim - 3);
  const auto& mat2_batch_sizes = mat2_sizes.narrow(1, 0, mat2_dim - 3);
  TORCH_CHECK(
      at::equal(self_batch_sizes, mat2_batch_sizes),
      "matmul: For nested tensors, batch dimensions must have the same sizes, ",
      "no broadcasting is currently performed. Got batch shapes for self ",
      self_batch_sizes,
      " and batch shapes for mat2 ",
      mat2_batch_sizes);
  // Ensure last dim of self and second last dim of mat2 have the same size
  const auto& self_dim_size = self_sizes.select(1, -1);
  const auto& mat2_dim_size = mat2_sizes.select(1, -2);
  TORCH_CHECK(
      at::equal(self_dim_size, mat2_dim_size),
      "matmul: Nested tensors cannot be matrix multiplied, last dimension of self has sizes",
      self_dim_size,
      "second last dimension of mat2 has sizes",
      mat2_dim_size);

  // use bmm inference-only fast path for [N, n_heads, *, head_dim] [N, n_heads,
  // head_dim, *]
  if (self.is_cuda() && self_dim == 4 && self.is_contiguous() &&
      mat2_dim == 4 && mat2.is_contiguous() &&
      !(GradMode::is_enabled() &&
        (self.requires_grad() || mat2.requires_grad()))) {
    const auto& self_opt_head_dim = self_ptr->opt_size(1);
    const auto& mat2_opt_head_dim = mat2_ptr->opt_size(1);
    if (self_opt_head_dim.has_value() && mat2_opt_head_dim.has_value() &&
        self_opt_head_dim.value() == mat2_opt_head_dim.value()) {
      return matmul_with_bmm_nested(self, mat2);
    }
  }

  // Construct output size from input sizes
  Tensor output_sizes = self_sizes.clone();
  // The last entry in every row of output_sizes should be last column of
  // mat2_sizes
  output_sizes.index_put_(
      {at::indexing::Slice(), -1}, mat2_sizes.select(1, -1).clone());

  auto self_padded = self_contig.to_padded_tensor(0.);
  auto mat2_padded = mat2_contig.to_padded_tensor(0.);
  auto output_padded = at::matmul(self_padded, mat2_padded);
  auto output_nested = nested_from_padded_generic(output_padded, output_sizes);
  return output_nested;
}

Tensor& matmul_out_nested(
    const Tensor& tensor1,
    const Tensor& tensor2,
    Tensor& result) {
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
    } else {
      sizes.push_back(-1);
    }
  }
  result.reshape(sizes);
  result.copy_(function_result);
  return result;
}

} // namespace native
} // namespace at
