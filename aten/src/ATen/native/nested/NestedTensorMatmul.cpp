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

Tensor matmul_with_bmm_nested(const Tensor& self, const Tensor& mat2) {
  // Tensor self = self_.contiguous();
  // Tensor mat2 = mat2_.contiguous();
  // self [N, n_heads, *, head_dim]
  // mat2 [N, n_heads, head_dim, *]
  const auto self_ptr = get_nested_tensor_impl(self);
  const auto mat2_ptr = get_nested_tensor_impl(mat2);
  // metadata for self
  std::vector<IntArrayRef> self_sizes = NestedTensor_get_sizes(self_ptr);
  std::vector<IntArrayRef> self_strides = NestedTensor_get_strides(self_ptr);
  std::vector<int64_t> self_offsets = self_ptr->get_storage_offsets();
  auto opt = self_ptr->get_nested_size_tensor().options();

  // metadata for mat2
  std::vector<IntArrayRef> mat2_sizes = NestedTensor_get_sizes(mat2_ptr);
  std::vector<IntArrayRef> mat2_strides = NestedTensor_get_strides(mat2_ptr);
  std::vector<int64_t> mat2_offsets = mat2_ptr->get_storage_offsets();
  auto opt2 = mat2_ptr->get_nested_size_tensor().options();

  int64_t N = self_sizes.size();
  int64_t n_heads = self_sizes[0][0];

  // viewed metadata for self
  auto self_new_sizes = at::empty({N * n_heads, 2}, opt);
  int64_t* self_new_sizes_ptr = self_new_sizes.data_ptr<int64_t>();

  auto self_new_strides = at::empty({N * n_heads, 2}, opt);
  int64_t* self_new_strides_ptr = self_new_strides.data_ptr<int64_t>();
  std::vector<int64_t> self_new_offsets;

  // viewed metadata for mat2
  auto mat2_new_sizes = at::empty({N * n_heads, 2}, opt2);
  int64_t* mat2_new_sizes_ptr = mat2_new_sizes.data_ptr<int64_t>();

  auto mat2_new_strides = at::empty({N * n_heads, 2}, opt2);
  int64_t* mat2_new_strides_ptr = mat2_new_strides.data_ptr<int64_t>();
  std::vector<int64_t> mat2_new_offsets;

  for (int64_t i = 0; i < N; i++) {
    const IntArrayRef& self_size_i = self_sizes[i];
    const IntArrayRef& self_stride_i = self_strides[i];
    int64_t self_offset = self_offsets[i];

    const IntArrayRef& mat2_size_i = mat2_sizes[i];
    const IntArrayRef& mat2_stride_i = mat2_strides[i];
    int64_t mat2_offset = mat2_offsets[i];
    for (int64_t j = 0; j < n_heads; j++) {
      auto idx = (i * n_heads + j) * 2;
      self_new_sizes_ptr[idx] = self_size_i[1];
      self_new_sizes_ptr[idx + 1] = self_size_i[2];
      self_new_strides_ptr[idx] = self_stride_i[1];
      self_new_strides_ptr[idx + 1] = self_stride_i[2];
      self_new_offsets.push_back(self_offset);
      self_offset += self_stride_i[0];

      mat2_new_sizes_ptr[idx] = mat2_size_i[1];
      mat2_new_sizes_ptr[idx + 1] = mat2_size_i[2];
      mat2_new_strides_ptr[idx] = mat2_stride_i[1];
      mat2_new_strides_ptr[idx + 1] = mat2_stride_i[2];
      mat2_new_offsets.push_back(mat2_offset);
      mat2_offset += mat2_stride_i[0];
    }
  }


  // view self as [N * n_heads, *, head_dim] (collapse first 2 dims)
  auto viewed_self = create_nested_view_tensor(
      self, self_new_sizes, self_new_strides, std::vector<int64_t>(self_new_offsets));

  // view mat2 as [N * n_heads, head_dim, *] (collapse first 2_dims)
  auto viewed_mat2 = create_nested_view_tensor(
      mat2, mat2_new_sizes, mat2_new_strides, std::vector<int64_t>(mat2_new_offsets));

  // output [N * n_heads, *, *]
  auto bmm_output = at::bmm(viewed_self, viewed_mat2);

  // generate metadata for viewing output as [N, n_heads, *, *]
  // output of bmm should be contiguous so stride calculations should hold
  auto out_new_sizes = at::empty({N, 3}, opt);
  auto out_new_strides = at::empty({N, 3}, opt);
  std::vector<int64_t> out_new_offsets;

  int64_t* out_new_sizes_ptr = out_new_sizes.data_ptr<int64_t>();
  int64_t* out_new_strides_ptr = out_new_strides.data_ptr<int64_t>();

  int64_t out_offset = 0;
  for (int64_t i = 0; i < N; i++) {
    out_new_offsets.push_back(out_offset);
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
      bmm_output, out_new_sizes, out_new_strides, std::vector<int64_t>(out_new_offsets));

  return viewed_out;

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

  // use bmm inference-only fast path for [N, n_heads, *, head_dim] [N, n_heads, head_dim, *]
  if (self.is_cuda() &&
      self_dim == 4 && self.is_contiguous() &&
      mat2_dim == 4 && mat2.is_contiguous() &&
      !(GradMode::is_enabled() && (self.requires_grad() || mat2.requires_grad()))) {
    auto n_heads = self_sizes.select(0, 1).select(0, 0).item<int64_t>();
    auto self_first_dim_n_heads = at::all(self_sizes.select(1, 0) == n_heads).item<bool>();
    auto mat2_first_dim_n_heads = at::all(mat2_sizes.select(1, 0) == n_heads).item<bool>();
    if (self_first_dim_n_heads && mat2_first_dim_n_heads) {
      return matmul_with_bmm_nested(self, mat2);
    }
  }

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

} // namespace native
} // namespace at
