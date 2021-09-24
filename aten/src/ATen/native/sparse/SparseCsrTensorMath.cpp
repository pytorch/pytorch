#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mkl/SparseBlasImpl.h>
#include <ATen/native/sparse/SparseBlasImpl.h>

#include <algorithm>

namespace at {
namespace meta {

TORCH_META_FUNC(_convert_indices_from_coo_to_csr) (
  const Tensor& self, const int64_t size, const bool out_int32
) {
  TORCH_CHECK(self.dim() <= 1, "Input is supposed to be a vector");
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  set_output(size + 1, options);
}

} // namespace meta

namespace {

constexpr int64_t GRAIN_SIZE = at::internal::GRAIN_SIZE;

template <typename input_t, typename output_t>
void convert_indices_from_coo_to_csr_cpu(const Tensor& result, const Tensor& input, const int64_t size) {
  int64_t numel = input.numel();
  const input_t* data_in = input.data_ptr<input_t>();
  output_t* data_out = result.data_ptr<output_t>();

  if (numel == 0) {
    result.zero_();
    return;
  }

  for (int64_t i = 0; i <= data_in[0]; i++)
    data_out[i] = static_cast<output_t>(0);

  at::parallel_for(0, numel - 1, GRAIN_SIZE, [&](int64_t start, int64_t end) {
    input_t curr_value = data_in[start], next_value;
    for (int64_t i = start; i < end; i++) {
      next_value = data_in[i + 1];
      for (; curr_value < next_value; curr_value++)
        data_out[curr_value + 1] = static_cast<output_t>(i + 1);
    }
  });

  for (int64_t i = data_in[numel - 1] + 1; i < size + 1; i++)
    data_out[i] = static_cast<output_t>(numel);
}

} // end anonymous namespace

namespace native {

using namespace at::sparse_csr;
// certain utiliy functions are usable from sparse COO.
using namespace at::sparse;

static constexpr bool is_mkl_supported() {
#ifdef _MSC_VER
  return false;
#elif  __APPLE__ || __MACH__
  return false;
#else
  return true;
#endif
}

// Only accept squares sparse matrices or dense input as a vector
// TODO: Check what happens with MKL, the output error reported with non square matrices tends to be high
// See: https://github.com/pytorch/pytorch/issues/58770
bool is_square_or_vec(int64_t dim_i, int64_t dim_j, int64_t dim_k) {
  return (dim_i == dim_k  && dim_k == dim_j) || (dim_i == dim_j && dim_k == 1);
}

template <typename scalar_t>
void addmm_out_sparse_csr_native_cpu(const Tensor& sparse, const Tensor& dense, const Tensor& r, Scalar alpha, Scalar beta) {

  auto dim_i = sparse.size(0);
  auto dim_j = sparse.size(1);
  auto dim_k = dense.size(1);
  auto nnz = sparse._nnz();

  auto csr = sparse.crow_indices();
  auto col_indices = sparse.col_indices();
  auto values = sparse.values();

  scalar_t cast_alpha = alpha.to<scalar_t>();
  scalar_t cast_beta = beta.to<scalar_t>();
  AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "csr_mm_crow_indices", [&]() {
    auto csr_accessor = csr.accessor<index_t, 1>();
    auto col_indices_accessor = col_indices.accessor<index_t, 1>();

    auto values_accessor = values.accessor<scalar_t, 1>();
    scalar_t* dense_ptr = dense.data_ptr<scalar_t>();
    scalar_t* r_ptr = r.data_ptr<scalar_t>();

    int64_t dense_stride0 = dense.stride(0);
    int64_t dense_stride1 = dense.stride(1);
    int64_t r_stride0 = r.stride(0);
    int64_t r_stride1 = r.stride(1);

    at::parallel_for(
        0,
        dim_i,
        internal::GRAIN_SIZE,
        [&](int64_t irow_start, int64_t irow_end) {
            for (index_t h = irow_start; h < irow_end; ++h) {
              index_t i_start = csr_accessor[h];
              index_t i_end = csr_accessor[h+1];
              for (index_t i = i_start; i < i_end; i++) {
                scalar_t val = values_accessor[i];
                index_t col = col_indices_accessor[i];
                at::native::cpublas::axpy<scalar_t>(dim_k,
                    cast_alpha * val,
                    dense_ptr + col * dense_stride0, dense_stride1,
                    r_ptr + h * r_stride0, r_stride1);
              }
            }
    });
  });
}

// Functions for matrix multiplication.
Tensor& addmm_out_sparse_csr_cpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  TORCH_INTERNAL_ASSERT(mat1.is_sparse_csr());

  // TODO: remove this, there are no codegenerated checks for devices yet
  TORCH_CHECK(
    !self.is_cuda(),
    "Expected all tensors to be on the same device. addmm expected 't' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(
      !result.is_cuda(),
      "Expected all tensors to be on the same device. addmm: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(
      !mat1.is_cuda(),
      "Expected all tensors to be on the same device. addmm: expected 'mat1' to be a CPU tensor, but got a CUDA tensor");
  TORCH_CHECK(
      !mat2.is_cuda(),
      "Expected all tensors to be on the same device. addmm: expected 'mat2' to be a CPU tensor, but got a CUDA tensor");

  // All the checks are from addmm_out_cuda_impl (ATen/native/cuda/Blas.cpp) and TORCH_META_FUNC(addmm) (ATen/native/LinearAlgebra.cpp)
  // TODO: remove code duplication and unify code
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");
  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
      mat1.sizes()[0], "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  IntArrayRef self__sizes;
  c10::MaybeOwned<Tensor> self_;
  if (&result != &self && self.layout() == kStrided) {
    self_ = expand_size(self, {mat1_sizes[0], mat2_sizes[1]}, "addmm");
    self__sizes = self_->sizes();
  } else {
    self_ = c10::MaybeOwned<Tensor>::borrowed(self);
    self__sizes = self_->sizes();
  }

  TORCH_CHECK(((self_->dim() == 2) && (self_->sizes()[0] == mat1.sizes()[0]) && (self_->sizes()[1] == mat2.sizes()[1])),
  "The input tensor must be a matrix with size ", mat1.sizes()[0], "x", mat2.sizes()[1], ", but got a ", self_->dim(),
  "-D tensor with size ", self__sizes[0], "x", self__sizes[1]);

  if (&result != &self) {
    if (result.layout() == kStrided) {
      at::native::resize_output(result, self__sizes);
    } else {
      at::native::resize_as_sparse_csr_(result, *self_);
    }
    result.copy_(*self_);
  }

  IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
  }

  if (mat1._nnz() == 0 && mat2.layout() == kStrided) {
    // According to docs, when beta==0 values in self should be ignored. nans and infs should not propagate
    if (beta.toComplexDouble() == 0.) {
      result.zero_();
    } else {
      result.mul_(beta);
    }
    return result;
  }

  if (mat2.is_sparse_csr() && (mat1._nnz() == 0 || mat2._nnz() == 0)) {
    if (beta.toComplexDouble() == 0.) {
      result.values().zero_();
    } else {
      result.values().mul_(beta);
    }
    return result;
  }

#if !AT_MKL_ENABLED()
    if (mat2.is_sparse_csr() && result.is_sparse_csr()) {
      TORCH_CHECK(false, "Calling addmm on sparse CPU tensors requires compiling PyTorch with MKL. Please use PyTorch built with MKL.");
    }
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.layout() == kStrided);
    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "addmm_sparse_dense", [&] {
        addmm_out_sparse_csr_native_cpu<scalar_t>(mat1, mat2, result, self_, alpha, beta);
    });
#else
  sparse::impl::mkl::addmm_out_sparse_csr(mat1, mat2, beta, alpha, result);
#endif
  return result;
}

Tensor addmm_sparse_csr_dense(
    const Tensor& self,
    const SparseCsrTensor& sparse,
    const Tensor& dense,
    const Scalar& beta,
    const Scalar& alpha) {
  Tensor r = at::empty({0, 0}, self.options());
  at::addmm_out(r, self, sparse, dense, beta, alpha);
  return r;
}

Tensor& _sparse_csr_mm_out(
    const Tensor& mat1,
    const Tensor& mat2,
    Tensor& result) {
  Tensor zero;
  if (result.is_sparse_csr()) {
    zero = at::empty({mat1.size(0), mat2.size(1)}, mat2.options());
  } else {
    zero = at::zeros({mat1.size(0), mat2.size(1)}, mat2.options());
  }
  return at::addmm_out(result, zero, mat1, mat2, 0.0, 1.0);
}

Tensor _sparse_csr_addmm(
    const Tensor& t,
    const SparseCsrTensor& sparse,
    const Tensor& dense,
    const Scalar& beta,
    const Scalar& alpha) {
  // _sparse_addmm forward is functionally equivalent to addmm; it's
  // just the backward that is different.  This technically does an
  // unnecessary redispatch, I was too lazy to make it not do that
  return at::addmm(t, sparse, dense, beta, alpha);
}

// Functions for element-wise addition.
Tensor add_sparse_csr(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  auto commonDtype = at::result_type(self, other);
  alpha_check(commonDtype, alpha);
  Tensor result = at::empty({0, 0}, self.options().dtype(commonDtype));
  return at::add_out(result, self, other, alpha); // redispatch!
}

Tensor& add_sparse_csr_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return at::add_out(self, self, other, alpha); // redispatch!
}

void add_out_dense_sparse_csr_cpu(
    const Tensor& out,
    const Tensor& dense,
    const SparseCsrTensor& src,
    const Scalar& alpha) {
  TORCH_INTERNAL_ASSERT(dense.layout() == kStrided);
  TORCH_INTERNAL_ASSERT(src.is_sparse_csr());
  TORCH_INTERNAL_ASSERT(dense.device() == kCPU);

  TORCH_CHECK(
      out.is_contiguous(),
      "out argument must be contiguous, but got: ",
      out.suggest_memory_format());
  TORCH_CHECK(
      out.device() == kCPU,
      "add: expected 'out' to be CPU tensor, but got tensor on device: ",
      out.device());
  TORCH_CHECK(
      src.device() == kCPU,
      "add: expected 'other' to be a CPU tensor, but got tensor on device: ",
      src.device());

  TORCH_CHECK(
      dense.sizes().equals(src.sizes()),
      "add: expected 'self' and 'other' to have same size, but self has size ",
      dense.sizes(),
      " while other has size ",
      src.sizes(),
      " (FYI: op2-sparse addition does not currently support broadcasting)");

  auto commonDtype = promoteTypes(dense.scalar_type(), src.scalar_type());
  TORCH_CHECK(
      canCast(commonDtype, out.scalar_type()),
      "Can't convert result type ",
      commonDtype,
      " to output ",
      out.scalar_type(),
      " in add operation");

  auto src_values = src.values();
  auto src_crow_indices = src.crow_indices();
  auto src_col_indices = src.col_indices();

  resize_output(out, dense.sizes());

  Tensor resultBuffer = out;
  Tensor valuesBuffer = src_values.to(commonDtype);

  if (out.scalar_type() != commonDtype) {
    resultBuffer = dense.to(commonDtype);
  } else if (!is_same_tensor(out, dense)) {
    resultBuffer.copy_(dense);
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBool, kBFloat16,
      commonDtype,
      "add_out_op2_sparse_csr",
      [&valuesBuffer, &resultBuffer, &alpha, &src_crow_indices, &src_col_indices]() {
        AT_DISPATCH_INDEX_TYPES(
            src_crow_indices.scalar_type(),
            "csr_add_out_crow_indices",
            [&valuesBuffer, &resultBuffer, &alpha, &src_crow_indices, &src_col_indices]() {
              auto values_accessor = valuesBuffer.accessor<scalar_t, 1>();
              scalar_t* out_ptr = resultBuffer.data_ptr<scalar_t>();
              scalar_t cast_value = alpha.to<scalar_t>();

              auto crow_indices_accessor =
                  src_crow_indices.accessor<index_t, 1>();
              auto col_indices_accessor =
                  src_col_indices.accessor<index_t, 1>();
              auto out_strides0 = resultBuffer.strides()[0];
              auto out_strides1 = resultBuffer.strides()[1];

              for (index_t irow = 0; irow < src_crow_indices.size(0) - 1;
                   ++irow) {
                index_t start_index = crow_indices_accessor[irow];
                index_t end_index = crow_indices_accessor[irow + 1];

                for (index_t i = start_index; i < end_index; ++i) {
                  auto icol = col_indices_accessor[i];
                  auto index = resultBuffer.storage_offset() + irow * out_strides0 +
                      icol * out_strides1;
                  out_ptr[index] += cast_value * values_accessor[i];
                }
              }
            });
      });
  if (out.scalar_type() != commonDtype) {
    out.copy_(resultBuffer);
  }
}

Tensor& add_out_sparse_csr_cpu(
    const Tensor& self,
    const SparseCsrTensor& other,
    const Scalar& alpha,
    SparseCsrTensor& out) {
  if (self.layout() == kStrided) {
    add_out_dense_sparse_csr_cpu(out, self, other, alpha);
  } else {
    TORCH_CHECK(
        self.sizes().equals(other.sizes()),
        "torch.add: Expected input tensors to have the same shape, but got tensor `self` with shape ",
        self.sizes(),
        " and tensor `other` with shape ",
        other.sizes());
    at::native::resize_as_sparse_csr_(out, self);
    sparse::impl::cpu::add_out_sparse_csr(self, other, alpha, out);
  }
  return out;
}

TORCH_IMPL_FUNC(_convert_indices_from_coo_to_csr_structured_cpu) (
  const Tensor& input, const int64_t size, const bool out_int32, const Tensor& result
) {
  if (out_int32) {
    AT_DISPATCH_INTEGRAL_TYPES(input.scalar_type(), "convert_indices_from_coo_to_csr_cpu", [&] {
      convert_indices_from_coo_to_csr_cpu<scalar_t, int>(result, input, size);
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(input.scalar_type(), "convert_indices_from_coo_to_csr_cpu", [&] {
      convert_indices_from_coo_to_csr_cpu<scalar_t, int64_t>(result, input, size);
    });
  }
}

} // namespace native
} // namespace at
