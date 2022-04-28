#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/mkl/Sparse.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mkl/SparseBlasImpl.h>
#include <ATen/native/sparse/SparseBlasImpl.h>
#include <ATen/native/sparse/SparseCsrTensorMath.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#else
#include <ATen/ops/_conj_physical_native.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_unique.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/add.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/angle.h>
#include <ATen/ops/angle_native.h>
#include <ATen/ops/asin.h>
#include <ATen/ops/asin_native.h>
#include <ATen/ops/asinh.h>
#include <ATen/ops/asinh_native.h>
#include <ATen/ops/atan.h>
#include <ATen/ops/atan_native.h>
#include <ATen/ops/atanh.h>
#include <ATen/ops/atanh_native.h>
#include <ATen/ops/ceil.h>
#include <ATen/ops/ceil_native.h>
#include <ATen/ops/conj_physical.h>
#include <ATen/ops/conj_physical_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/erf.h>
#include <ATen/ops/erf_native.h>
#include <ATen/ops/erfinv.h>
#include <ATen/ops/erfinv_native.h>
#include <ATen/ops/expm1.h>
#include <ATen/ops/expm1_native.h>
#include <ATen/ops/floor.h>
#include <ATen/ops/floor_native.h>
#include <ATen/ops/isinf.h>
#include <ATen/ops/isinf_native.h>
#include <ATen/ops/isnan.h>
#include <ATen/ops/isnan_native.h>
#include <ATen/ops/isneginf.h>
#include <ATen/ops/isneginf_native.h>
#include <ATen/ops/isposinf.h>
#include <ATen/ops/isposinf_native.h>
#include <ATen/ops/log1p.h>
#include <ATen/ops/log1p_native.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/neg_native.h>
#include <ATen/ops/normal_native.h>
#include <ATen/ops/rad2deg.h>
#include <ATen/ops/rad2deg_native.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/round.h>
#include <ATen/ops/round_native.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/sgn.h>
#include <ATen/ops/sgn_native.h>
#include <ATen/ops/sign.h>
#include <ATen/ops/sign_native.h>
#include <ATen/ops/signbit.h>
#include <ATen/ops/signbit_native.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/sin_native.h>
#include <ATen/ops/sinh.h>
#include <ATen/ops/sinh_native.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/sqrt_native.h>
#include <ATen/ops/tan.h>
#include <ATen/ops/tan_native.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/tanh_native.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/trunc.h>
#include <ATen/ops/trunc_native.h>
#include <ATen/ops/zero_native.h>
#include <ATen/ops/zeros.h>
#endif

#include <algorithm>

namespace at {
namespace meta {

TORCH_META_FUNC(_convert_indices_from_coo_to_csr)
(const Tensor& self, const int64_t size, const bool out_int32) {
  TORCH_CHECK(self.dim() <= 1, "Input is supposed to be a vector");
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options =
      TensorOptions().device(self.options().device()).dtype(scalar_type);
  set_output(size + 1, options);
}

TORCH_META_FUNC(_convert_indices_from_csr_to_coo)
(const Tensor& crow_indices,
 const Tensor& col_indices,
 const bool out_int32,
 const bool transpose) {
  TORCH_CHECK(
      crow_indices.dim() == 1, "crow_indices is supposed to be a vector");
  TORCH_CHECK(col_indices.dim() == 1, "col_indices is supposed to be a vector");
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = crow_indices.options().dtype(scalar_type);
  set_output(0, {2, col_indices.numel()}, {}, options, {});
}

} // namespace meta

namespace {

constexpr int64_t GRAIN_SIZE = at::internal::GRAIN_SIZE;

template <typename input_t, typename output_t>
void convert_indices_from_coo_to_csr_cpu(
    const Tensor& result,
    const Tensor& input,
    const int64_t size) {
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
    for (const auto i : c10::irange(start, end)) {
      next_value = data_in[i + 1];
      for (; curr_value < next_value; curr_value++)
        data_out[curr_value + 1] = static_cast<output_t>(i + 1);
    }
  });

  for (int64_t i = data_in[numel - 1] + 1; i < size + 1; i++)
    data_out[i] = static_cast<output_t>(numel);
}

template <typename F>
Tensor& unary_op_out(F op_out, const Tensor& self, Tensor& result) {
  TORCH_INTERNAL_ASSERT(self.is_sparse_csr());
  TORCH_INTERNAL_ASSERT(result.is_sparse_csr());

  if (!result.is_same(self)) {
    // For the case of (0x0) result tensor, manually resize `result` tensor
    // to the size of `self` tensor
    if (result.numel() == 0) {
      at::native::resize_as_sparse_csr_(result, self);
    }
    // copy_sparse_csr_ internally checks the sizes of result and self tensors
    // Hence no external size check required
    at::native::copy_sparse_csr_(result, self);
  }

  auto self_values = self.values();
  auto result_values = result.values();

  op_out(self_values, result_values);
  return result;
}

template <typename F, typename... Args>
Tensor& unary_op_inplace(Tensor& self, const F& op_inplace, Args&&... args) {
  TORCH_INTERNAL_ASSERT(self.is_sparse_csr());

  auto self_values = self.values();
  (self_values.*op_inplace)(std::forward<Args>(args)...);
  return self;
}

template <typename input_t, typename output_t>
void convert_indices_from_csr_to_coo_cpu(
    const Tensor& indices,
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const bool transpose = false) {
  int64_t nrows = crow_indices.numel() - 1;
  if (nrows == 0) {
    indices.zero_();
    return;
  }
  auto crow_indices_ = crow_indices.expect_contiguous();
  const input_t* crow_indices_data_in = crow_indices_->data_ptr<input_t>();
  TORCH_INTERNAL_ASSERT(indices.is_contiguous());
  auto row0 = indices.select(0, transpose ? 1 : 0);
  auto row1 = indices.select(0, transpose ? 0 : 1);
  output_t* data_out = row0.data_ptr<output_t>();
  row1.copy_(*col_indices.expect_contiguous());
  at::parallel_for(0, nrows, GRAIN_SIZE, [&](int64_t start, int64_t end) {
    for (const auto i : c10::irange(start, end)) {
      std::fill(
          &data_out[crow_indices_data_in[i]],
          &data_out[crow_indices_data_in[i + 1]],
          static_cast<output_t>(i));
    }
  });
}

} // end anonymous namespace

namespace native {

using namespace at::sparse_csr;
// certain utiliy functions are usable from sparse COO.
using namespace at::sparse;

namespace {

template <typename F>
inline Tensor get_result_tensor_for_unary_op(F op, const Tensor& input) {
  auto values = input.values();

  // To handle type promotion for inputs to unary ops,
  // we first get the result from the underlined op, and use the result
  // to create a sparse CSR tensor, which is used as the input to the out=
  // variant
  auto result_values = op(values);

  auto result = at::native::_sparse_csr_tensor_unsafe(
      input.crow_indices().clone(),
      input.col_indices().clone(),
      result_values,
      input.sizes(),
      result_values.scalar_type(),
      input.layout(),
      result_values.device());

  return result;
}
} // namespace

static constexpr bool is_mkl_supported() {
#ifdef _MSC_VER
  return false;
#elif __APPLE__ || __MACH__
  return false;
#else
  return true;
#endif
}

// Only accept squares sparse matrices or dense input as a vector
// TODO: Check what happens with MKL, the output error reported with non square
// matrices tends to be high See:
// https://github.com/pytorch/pytorch/issues/58770
bool is_square_or_vec(int64_t dim_i, int64_t dim_j, int64_t dim_k) {
  return (dim_i == dim_k && dim_k == dim_j) || (dim_i == dim_j && dim_k == 1);
}

Tensor& normal_sparse_csr_(
    Tensor& self,
    double mean,
    double std,
    c10::optional<Generator> gen) {
  return unary_op_inplace(self, &Tensor::normal_, mean, std, gen);
}

/* Implementation of Unary Ufuncs, those supported for Sparse CSR Layout
 * Only simple funcs, with 0->0 correspondence are currently supported. */

#define CREATE_UNARY_UFUNC_OUT(op_name)                                  \
  Tensor& op_name##_sparse_csr_out(const Tensor& self, Tensor& result) { \
    return unary_op_out(&at::op_name##_outf, self, result);              \
  }

#define CREATE_UNARY_UFUNC_FUNCTIONAL(op_name)                 \
  Tensor op_name##_sparse_csr(const Tensor& self) {            \
    return get_result_tensor_for_unary_op(&at::op_name, self); \
  }

#define CREATE_UNARY_UFUNC_INPLACE(op_name)             \
  Tensor& op_name##_sparse_csr_(Tensor& self) {         \
    return unary_op_inplace(self, &Tensor::op_name##_); \
  }

#define CREATE_UNARY_UFUNC(op_name)       \
  CREATE_UNARY_UFUNC_OUT(op_name);        \
  CREATE_UNARY_UFUNC_FUNCTIONAL(op_name); \
  CREATE_UNARY_UFUNC_INPLACE(op_name);

#define CREATE_UNARY_UFUNC_NO_INPLACE(op_name) \
  CREATE_UNARY_UFUNC_OUT(op_name);             \
  CREATE_UNARY_UFUNC_FUNCTIONAL(op_name);

// Exhaustive list of the unary ufuncs supported by sparse CSR
CREATE_UNARY_UFUNC(abs);
CREATE_UNARY_UFUNC(asin);
CREATE_UNARY_UFUNC(asinh);
CREATE_UNARY_UFUNC(atan);
CREATE_UNARY_UFUNC(atanh);
CREATE_UNARY_UFUNC(ceil);
CREATE_UNARY_UFUNC(erf);
CREATE_UNARY_UFUNC(erfinv);
CREATE_UNARY_UFUNC(expm1);
CREATE_UNARY_UFUNC(floor);
CREATE_UNARY_UFUNC(log1p);
CREATE_UNARY_UFUNC(neg);
CREATE_UNARY_UFUNC(rad2deg);
CREATE_UNARY_UFUNC(sign);
CREATE_UNARY_UFUNC(sin);
CREATE_UNARY_UFUNC(sinh);
CREATE_UNARY_UFUNC(sgn);
CREATE_UNARY_UFUNC(sqrt);
CREATE_UNARY_UFUNC(tan);
CREATE_UNARY_UFUNC(tanh);
CREATE_UNARY_UFUNC(trunc);
CREATE_UNARY_UFUNC(conj_physical);

CREATE_UNARY_UFUNC_INPLACE(zero);

// With addition of `round.decimals` overload, using CREATE_UNARY_UFUNC leads
// to unresolved overload.
Tensor& round_sparse_csr_out(const Tensor& self, Tensor& result) {
  return unary_op_out(&at::_ops::round_out::call, self, result);
}

Tensor round_sparse_csr(const Tensor& self) {
  return get_result_tensor_for_unary_op(&at::_ops::round::call, self);
}

Tensor& round_sparse_csr_(Tensor& self) {
  TORCH_INTERNAL_ASSERT(self.is_sparse_csr());
  self.values().round_();
  return self;
}

// angle, isneginf, isposinf and signbit currently don't have an inplace variant
CREATE_UNARY_UFUNC_NO_INPLACE(angle);
CREATE_UNARY_UFUNC_NO_INPLACE(isneginf);
CREATE_UNARY_UFUNC_NO_INPLACE(isposinf);
CREATE_UNARY_UFUNC_NO_INPLACE(signbit);

// isnan and isinf don't have an out variant
CREATE_UNARY_UFUNC_FUNCTIONAL(isnan);
CREATE_UNARY_UFUNC_FUNCTIONAL(isinf);

template <typename scalar_t>
void addmm_out_sparse_csr_native_cpu(
    const Tensor& sparse,
    const Tensor& dense,
    const Tensor& r,
    Scalar alpha,
    Scalar beta) {
  auto dim_i = sparse.size(0);
  auto dim_k = dense.size(1);

  auto csr = sparse.crow_indices();
  auto col_indices = sparse.col_indices();
  auto values = sparse.values();

  scalar_t cast_alpha = alpha.to<scalar_t>();
  r.mul_(beta);
  AT_DISPATCH_INDEX_TYPES(
      col_indices.scalar_type(), "csr_mm_crow_indices", [&]() {
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
                index_t i_end = csr_accessor[h + 1];
                for (index_t i = i_start; i < i_end; i++) {
                  scalar_t val = values_accessor[i];
                  index_t col = col_indices_accessor[i];
                  at::native::cpublas::axpy<scalar_t>(
                      dim_k,
                      cast_alpha * val,
                      dense_ptr + col * dense_stride0,
                      dense_stride1,
                      r_ptr + h * r_stride0,
                      r_stride1);
                }
              }
            });
      });
}

// Functions for matrix multiplication.
// result = beta * self + alpha (mat1 @ mat2)
Tensor& addmm_out_sparse_csr_cpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  switch (mat1.layout()) {
  case kStrided:
  case kSparseCsr:
    break;
  case kSparseCsc:
  case kSparseBsr:
  case kSparseBsc:
    TORCH_CHECK(false, "addmm_out_sparse_csr_cpu: layout ", mat1.layout(), " is not yet supported");
    break;
  default:
    TORCH_CHECK(false, "addmm_out_sparse_csr_cpu: layout ", mat1.layout(), " is not supported");
  }
// TODO: remove this, there are no codegenerated checks for devices yet
  sparse::impl::_check_is_cpu(self, "self");
  sparse::impl::_check_is_cpu(mat1, "mat1");
  sparse::impl::_check_is_cpu(mat2, "mat2");
  sparse::impl::_check_is_cpu(result, "result");

  // All the checks are from addmm_out_cuda_impl (ATen/native/cuda/Blas.cpp) and
  // TORCH_META_FUNC(addmm) (ATen/native/LinearAlgebra.cpp)
  // TODO: remove code duplication and unify code
  sparse::impl::_check_dim(mat1, 2, "mat1");
  sparse::impl::_check_dim(mat2, 2, "mat2");

  TORCH_CHECK(
      mat1.size(1) == mat2.size(0), "mat1 and mat2 shapes cannot be multiplied (",
      mat1.size(0), "x", mat1.size(1), " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

  c10::MaybeOwned<at::Tensor> self_;
  // Don't expand self if this is an in-place operation
  if (&result == &self) {
     self_ = c10::MaybeOwned<Tensor>::borrowed(self);
  } else {
     self_ = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm");
  }


  TORCH_CHECK(((self_->dim() == 2) &&
               (self_->size(0) == mat1.size(0)) &&
               (self_->size(1) == mat2.size(1))),
              "The input tensor must be a matrix with size ",
              mat1.size(0),
              "x",
              mat2.size(1),
              ", but got a ",
              self_->dim(),
              "-D tensor with size ",
              self_->size(0),
              "x",
              self_->size(1));

  if (&result != &self) {
    if (result.layout() == kStrided) {
      at::native::resize_output(result, self_->sizes());
    } else {
      result.resize_as_sparse_(*self_);
    }
    result.copy_(*self_);
  }

  if (result.numel() == 0) {
    return result;
  }

  if (sparse::impl::_is_sparse_and_zero(mat1) || sparse::impl::_is_sparse_and_zero(mat2)) {
    // According to docs, when beta==0 values in self should be ignored.
    // nans and infs should not propagate
    if (beta.toComplexDouble() == 0.) {
      result.zero_();
    } else {
      result.mul_(beta);
    }
    return result;
  }

#if !AT_USE_MKL_SPARSE()
  TORCH_CHECK(
      (mat1.is_sparse_csr() ||
       (mat2.is_sparse_csr() && result.is_sparse_csr())),
      false,
      "Calling addmm on sparse CPU tensors requires Linux platform. ",
      "Please use PyTorch built with MKL on Linux.");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.layout() == kStrided);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "addmm_sparse_dense", [&] {
        addmm_out_sparse_csr_native_cpu<scalar_t>(
            mat1, mat2, result, alpha, beta);
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
    // TODO: replace with at::zeros when it's implemented for sparse csr
    zero = at::empty({mat1.size(0), mat2.size(1)}, mat2.options());
  } else {
    zero = at::zeros({mat1.size(0), mat2.size(1)}, mat2.options());
  }
  return at::addmm_out(result, zero, mat1, mat2, 0.0, 1.0);
}

Tensor _sparse_csr_mm(const Tensor& mat1, const Tensor& mat2) {
  if (mat1.is_sparse_csr() && mat2.is_sparse_csr()) {
    // Return sparse
    // TODO: replace with at::zeros when it's implemented for sparse csr
    return at::addmm(
        at::empty({mat1.size(0), mat2.size(1)}, mat2.options()),
        mat1,
        mat2,
        0.0,
        1.0);
  }
  if (mat1.is_sparse_csr() && mat2.layout() == c10::kStrided) {
    // Return dense
    return at::addmm(
        at::zeros({mat1.size(0), mat2.size(1)}, mat2.options()),
        mat1,
        mat2,
        0.0,
        1.0);
  }
  if (mat1.layout() == c10::kStrided && mat2.is_sparse_csr()) {
    // Return dense
    return at::addmm(
        at::zeros({mat1.size(0), mat2.size(1)}, mat1.options()),
        mat1,
        mat2,
        0.0,
        1.0);
  }
  TORCH_INTERNAL_ASSERT(false, "Shouldn't get here. Please open an issue.");
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
Tensor add_sparse_csr(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  auto commonDtype = at::result_type(self, other);
  alpha_check(commonDtype, alpha);
  Tensor result = at::empty({0, 0}, self.options().dtype(commonDtype));
  return at::add_out(result, self, other, alpha); // redispatch!
}

Tensor& add_sparse_csr_(
    Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
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

  resize_output(out, dense.sizes());

  Tensor resultBuffer = out;

  if (out.scalar_type() != commonDtype) {
    resultBuffer = dense.to(commonDtype);
  } else if (!is_same_tensor(out, dense)) {
    resultBuffer.copy_(dense);
  }

  if (src._nnz() == 0) {
    return;
  }

  auto valuesBuffer = src_values.to(commonDtype).view({-1, src_values.size(-1)});
  resultBuffer = resultBuffer.view({-1, out.size(-2), out.size(-1)});
  auto src_crow_indices = src.crow_indices().view({-1, src.crow_indices().size(-1)});
  auto src_col_indices = src.col_indices().view({-1, src.col_indices().size(-1)});

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf,
      kBool,
      kBFloat16,
      commonDtype,
      "add_out_op2_sparse_csr",
      [&valuesBuffer,
       &resultBuffer,
       &alpha,
       &src_crow_indices,
       &src_col_indices]() {
        AT_DISPATCH_INDEX_TYPES(
            src_crow_indices.scalar_type(),
            "csr_add_out_crow_indices",
            [&valuesBuffer,
             &resultBuffer,
             &alpha,
             &src_crow_indices,
             &src_col_indices]() {
              auto batch_count = resultBuffer.dim() > 2 ? resultBuffer.size(-3) : 1;
              auto values_accessor = valuesBuffer.accessor<scalar_t, 2>();
              scalar_t* out_ptr = resultBuffer.data_ptr<scalar_t>();
              scalar_t cast_value = alpha.to<scalar_t>();

              auto crow_indices_accessor =
                  src_crow_indices.accessor<index_t, 2>();
              auto col_indices_accessor =
                  src_col_indices.accessor<index_t, 2>();
              auto out_strides = resultBuffer.strides();

              for (const auto batch_idx : c10::irange(batch_count)) {
                for (const auto irow : c10::irange(src_crow_indices.size(-1) - 1)) {
                  index_t start_index = crow_indices_accessor[batch_idx][irow];
                  index_t end_index = crow_indices_accessor[batch_idx][irow + 1];
                  for (const auto i : c10::irange(start_index, end_index)) {
                    auto icol = col_indices_accessor[batch_idx][i];
                    auto index = batch_idx * out_strides[0] + irow * out_strides[1] + icol * out_strides[2];
                    out_ptr[index] += cast_value * values_accessor[batch_idx][i];
                  }
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

TORCH_IMPL_FUNC(_convert_indices_from_coo_to_csr_structured_cpu)
(const Tensor& input,
 const int64_t size,
 const bool out_int32,
 const Tensor& result) {
  if (out_int32) {
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "convert_indices_from_coo_to_csr_cpu", [&] {
          convert_indices_from_coo_to_csr_cpu<scalar_t, int>(
              result, input, size);
        });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "convert_indices_from_coo_to_csr_cpu", [&] {
          convert_indices_from_coo_to_csr_cpu<scalar_t, int64_t>(
              result, input, size);
        });
  }
}

TORCH_IMPL_FUNC(_convert_indices_from_csr_to_coo_structured_cpu)
(const Tensor& crow_indices,
 const Tensor& col_indices,
 const bool out_int32,
 const bool transpose,
 const Tensor& result) {
  if (out_int32) {
    AT_DISPATCH_INTEGRAL_TYPES(
        crow_indices.scalar_type(), "convert_indices_from_csr_to_coo_cpu", [&] {
          convert_indices_from_csr_to_coo_cpu<scalar_t, int32_t>(
              result, crow_indices, col_indices, transpose);
        });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(
        crow_indices.scalar_type(), "convert_indices_from_csr_to_coo_cpu", [&] {
          convert_indices_from_csr_to_coo_cpu<scalar_t, int64_t>(
              result, crow_indices, col_indices, transpose);
        });
  }
}

/*
 * Based on
 * https://github.com/scipy/scipy/blob/8a64c938ddf1ae4c02a08d2c5e38daeb8d061d38/scipy/sparse/sparsetools/csr.h
 */
template <class I, class T>
void _csr_to_block_csr_cpu_kernel(
    const I n_row,
    const I n_col,
    const I R,
    const I C,
    const I* input_crow_indices,
    const I* input_col_indices,
    const T* input_values,
    I* result_crow_indices,
    I* result_col_indices,
    T* result_values) {
  // All blocks are possible, that is, may be allocated if a single non-zero
  // value lives within them. Otherwise they're not.

  // Allocate pointers for all possible column blocks plus 1
  std::vector<T*> blocks(n_col / C + 1, (T*)0);

  assert(n_row % R == 0);
  assert(n_col % C == 0);

  // Major assumptions
  // 1. Blocks must be square

  // Number of blocks along rows
  I n_brow = n_row / R;
  // Number of blocks along columns
  // I n_bcol = n_col / C;

  // Number of elements per block
  I RC = R * C;
  // Number of blocks overall
  I n_blks = 0;

  result_crow_indices[0] = 0;

  // Iterate over blocks along rows
  for (I block_i = 0; block_i < n_brow; block_i++) {
    // Iterate over rows within block
    for (I r = 0; r < R; r++) {
      I i = R * block_i + r; // row index
      for (I jj = input_crow_indices[i]; jj < input_crow_indices[i + 1]; jj++) {
        I j = input_col_indices[jj]; // column index

        // Block corresponding to column index
        I block_j = j / C;
        // Column within block
        I c = j % C;

        if (blocks[block_j] == 0) {
          blocks[block_j] = result_values + RC * n_blks;
          result_col_indices[n_blks] = block_j;
          n_blks++;
        }

        // Specific blocks entries should not be visited more than once.
        // Scipy code does an addition here. Why?
        *(blocks[block_j] + C * r + c) = input_values[jj];
      }
    }

    for (I jj = input_crow_indices[R * block_i];
         jj < input_crow_indices[R * (block_i + 1)];
         jj++) {
      blocks[input_col_indices[jj] / C] = 0;
    }

    result_crow_indices[block_i + 1] = n_blks;
  }
}

/*
 * Based on
 * https://github.com/scipy/scipy/blob/8a64c938ddf1ae4c02a08d2c5e38daeb8d061d38/scipy/sparse/sparsetools/csr.h
 */
template <class I>
I csr_count_blocks(
    const I n_row,
    const I n_col,
    const I R,
    const I C,
    const I Ap[],
    const I Aj[]) {
  std::vector<I> mask(n_col / C + 1, -1);
  I n_blks = 0;
  for (I i = 0; i < n_row; i++) {
    I bi = i / R;
    for (I jj = Ap[i]; jj < Ap[i + 1]; jj++) {
      I bj = Aj[jj] / C;
      if (mask[bj] != bi) {
        mask[bj] = bi;
        n_blks++;
      }
    }
  }
  return n_blks;
}

Tensor _csr_to_block_csr_cpu(const Tensor& self, IntArrayRef blocksize) {
  TORCH_CHECK(
      blocksize[0] == blocksize[1],
      "blocks must be square. ",
      "Got (",
      blocksize[0],
      ", ",
      blocksize[1],
      ") instead.");
  TORCH_CHECK(
      self.size(0) % blocksize[0] == 0 && self.size(1) % blocksize[1] == 0,
      "Block sparse CSR Tensors must have a size that is an ",
      "integral multiple of their block size. ",
      "Got Tensor of size (",
      self.size(0),
      ", ",
      self.size(1),
      ") with block size (",
      blocksize[0],
      ", ",
      blocksize[1],
      ") instead.");
  Tensor input_values = self.values().contiguous();
  Tensor input_crow_indices = self.crow_indices().contiguous();
  Tensor input_col_indices = self.col_indices().contiguous();

  // First we determine the number of blocks needed. For each given block, if it
  // contains a non-zero element we will allocate values and indices for it.
  int64_t num_blocks;
  int64_t n_row = self.size(0);
  int64_t n_col = self.size(1);
  AT_DISPATCH_INDEX_TYPES(
      input_crow_indices.scalar_type(), "_csr_to_block_csr_cpu", [&] {
        num_blocks = csr_count_blocks<index_t>(
            n_row,
            n_col,
            blocksize[0],
            blocksize[1],
            input_crow_indices.data_ptr<index_t>(),
            input_col_indices.data_ptr<index_t>());
      });

  Tensor result_values =
      input_values.new_zeros({num_blocks, blocksize[0], blocksize[1]});
  Tensor result_crow_indices =
      input_crow_indices.new_empty({(n_row / blocksize[0]) + 1});
  Tensor result_col_indices = input_col_indices.new_empty({num_blocks});

  // Next we copy over non-zero elements into the allocated blocks.
  AT_DISPATCH_INDEX_TYPES(
      input_crow_indices.scalar_type(), "_csr_to_block_csr_cpu", [&] {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
            input_values.scalar_type(), "_csr_to_block_csr_cpu", [&] {
              _csr_to_block_csr_cpu_kernel<index_t, scalar_t>(
                  n_row,
                  n_col,
                  blocksize[0],
                  blocksize[1],
                  input_crow_indices.data_ptr<index_t>(),
                  input_col_indices.data_ptr<index_t>(),
                  input_values.data_ptr<scalar_t>(),
                  result_crow_indices.data_ptr<index_t>(),
                  result_col_indices.data_ptr<index_t>(),
                  result_values.data_ptr<scalar_t>());
            });
      });
  return at::native::_sparse_csr_tensor_unsafe(
      result_crow_indices,
      result_col_indices,
      result_values,
      self.sizes(),
      result_values.scalar_type(),
      self.layout(),
      result_values.device());
}

Tensor _csr_to_block_csr(const Tensor& self, IntArrayRef blocksize) {
  Tensor self_values = self.values();
  Tensor self_crow_indices = self.crow_indices();
  Tensor self_col_indices = self.col_indices();
  Tensor cpu_result = _csr_to_block_csr_cpu(
      _sparse_csr_tensor_unsafe(self_crow_indices.cpu(),
                                self_col_indices.cpu(),
                                self_values.cpu(),
                                self.sizes(),
                                self_values.scalar_type(),
                                self.layout(),
                                self_values.device()),
      blocksize);
  Tensor result_values = cpu_result.values().to(self_values.options());
  Tensor result_crow_indices = cpu_result.crow_indices().to(self_crow_indices.options());
  Tensor result_col_indices = cpu_result.col_indices().to(self_col_indices.options());
  return at::native::_sparse_csr_tensor_unsafe(
      result_crow_indices,
      result_col_indices,
      result_values,
      self.sizes(),
      result_values.scalar_type(),
      self.layout(),
      result_values.device());
}

/*
    Reductions on sparse CSR tensors using masked semantics.

    - A CSR tensor is a 2D tensor that is specified by a 3-tuple
      (crow_indices, col_indices, values).

    - To support a reduction operator on a CSR tensor, define:

template <typename scalar_t>
struct Reduction...Op {
  inline scalar_t operator()(const scalar_t& a, const scalar_t& b) const {
    return a ... b;
  }
  inline scalar_t identity() const { return ...; }
};

Tensor _sparse_csr_..._cpu(const Tensor& input, IntArrayRef dims_to_sum, bool keepdim, c10::optional<ScalarType> dtype) {
  ...
      result = reduce_sparse_csr_cpu_template<scalar_t>(input_, dims_to_sum, keepdim, Reduction...Op<scalar_t>());
  ...
  return result;
}

      and add the following

        - func: _sparse_csr_op.dim_dtype(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
          dispatch:
            SparseCsrCUDA: _sparse_csr_..._cpu

      to native_functions.yaml

      Use ReductionAddOp and _sparse_csr_sum implementation as an example.

    - Since a CSR tensor dimensionality is always 2, only reductions
      with keepdim=True can be supported.

*/

namespace {

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_dim0_cpu_template(const Tensor& sparse, ReductionOp rop) {
  /*
    Consider the following sparse tensor:

    1 * * * *
    * * * 2 *
    * * 3 * *
    * * * * *
    4 * 5 * *

    that has CSR representation

      crow_indices = [0, 1, 2, 3, 3, 5]
      col_indices = [0, 3, 2, 0, 2]
      values = [1, 2, 3, 4, 5]

    Reduction with dim=0 results:

    rop(1,4) * rop(3,5) 2 *

    that has CSR representation

      new_crow_indices = [0, 3]
      new_col_indices = [0, 2, 3]
      new_values = [rop(1, 4], rop(3, 5), 2]

    In general, the CSR representation data can be computed as follows:

      new_col_indices, col_map = col_indices.unique(sorted=True, return_inverse=True)
      nnz = new_col_indices.numel()
      new_crow_indices = [0, nnz]
      new_values.resize(nnz); new_values.fill_(identity)
      for i in range(col_indices.numel()):
          new_values[col_map[i]] = rop(new_values[col_map[i], values[i])
   */

  Tensor col_indices = sparse.col_indices();
  Tensor values = sparse.values();
  auto numel = values.numel();
  Tensor new_col_indices;
  Tensor columns_map;

  /*
    Calling at::_unique constitutes the main bottleneck of this
    function. However, it is still about 5x faster than using the
    invariant:
      csr.sum(dim=0) == csr.transpose(0, 1).sum(dim=1)
  */
  std::tie(new_col_indices, columns_map) = at::_unique(col_indices, true, true);
  auto nnz = new_col_indices.numel();

  Tensor new_crow_indices = at::empty({2}, col_indices.options());
  new_crow_indices[0] = 0;
  new_crow_indices[1] = nnz;

  Tensor new_values = at::empty({nnz}, values.options());
  new_values.fill_(rop.identity());

  AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "reduce_sparse_csr_dim0_cpu_indices",
                          [&]() {
                            index_t* columns_map_ptr = columns_map.data_ptr<index_t>();
                            scalar_t* values_ptr = values.data_ptr<scalar_t>();
                            scalar_t* new_values_ptr = new_values.data_ptr<scalar_t>();

                            // There is no point in parallelizing the following for-loop
                            // because about 99.3% of the computation time is spent in the
                            // at::_unique call above.
                            for (int64_t i=0; i<numel; i++) {
                              index_t col = columns_map_ptr[i];
                              scalar_t val = values_ptr[i];
                              new_values_ptr[col] = rop(new_values_ptr[col], val);
                            }
                          });
  return at::native::_sparse_csr_tensor_unsafe(new_crow_indices, new_col_indices, new_values,
                                               {1, sparse.size(1)},
                                               new_values.scalar_type(),
                                               sparse.layout(),
                                               new_values.device());
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_dim1_cpu_template(const Tensor& sparse, ReductionOp rop) {
  /*
    Consider the following sparse tensor:

    1 * * * *
    * * * 2 *
    * * 3 * *
    * * * * *
    4 * 5 * *

    that has CSR representation

      crow_indices = [0, 1, 2, 3, 3, 5]
      col_indices = [0, 3, 2, 0, 2]
      values = [1, 2, 3, 4, 5]

    Reduction with dim=1 results:

    1
    2
    3
    *
    rop(4, 5)

    that has CSR representation

      new_crow_indices = [0, 1, 2, 3, 3, 4]
      new_col_indices = [0, 0, 0, 0]
      new_values = [1, 2, 3, rop(4, 5)]

    In general, the result CSR data can be computed as follows:

      new_crow_indices = [0]
      for i in range(1, nrows+1):
          new_crow_indices[i] = new_crow_indices[i-1] + (crow_indices[i] == crow_indices[i-1])
      nnz = new_crow_indices[-1]
      new_col_indices = zeros(nnz)
      new_values.resize(nnz)
      j = -1
      for i in range(1, nrows+1):
          if crow_indices[i] == crow_indices[i-1]:
              continue
          j += 1
          new_values[j] = rop(values[crow_indices[i] : crow_indices[i-1]])
  */

  Tensor crow_indices = sparse.crow_indices();
  auto ioptions = crow_indices.options();
  Tensor values = sparse.values();
  auto nrows = sparse.size(0);

  Tensor new_crow_indices = at::empty({crow_indices.numel()}, ioptions);
  Tensor new_col_indices = at::empty({}, ioptions);
  Tensor new_values = at::empty({}, values.options());
  Tensor row_map = at::empty({nrows}, ioptions);

  AT_DISPATCH_INDEX_TYPES(crow_indices.scalar_type(), "reduce_sparse_csr_dim1_cpu_indices",
                          [&]() {
    index_t* crow_indices_ptr = crow_indices.data_ptr<index_t>();
    index_t* new_crow_indices_ptr = new_crow_indices.data_ptr<index_t>();
    index_t* row_map_ptr = row_map.data_ptr<index_t>();
    int64_t nnz = 0;
    new_crow_indices_ptr[0] = 0;
    for(int64_t i=0; i<nrows; i++) {
      if (crow_indices_ptr[i] != crow_indices_ptr[i + 1]) {
        row_map_ptr[i] = nnz;
        nnz++;
      }
      new_crow_indices_ptr[i + 1] = nnz;
    }
    new_col_indices.resize_(nnz);
    new_col_indices.fill_(index_t(0));
    new_values.resize_(nnz);

    scalar_t* values_ptr = values.data_ptr<scalar_t>();
    scalar_t* new_values_ptr = new_values.data_ptr<scalar_t>();

    at::parallel_for(
        0,
        nrows,
        internal::GRAIN_SIZE,
        [&](int64_t irow_start, int64_t irow_end) {
            index_t i_end = crow_indices_ptr[irow_start];
            for (index_t h = irow_start; h < irow_end; ++h) {
              index_t i_start = i_end;
              i_end = crow_indices_ptr[h+1];
              if (i_start != i_end) {
                scalar_t res = values_ptr[i_start];
                for (index_t i = i_start + 1; i < i_end; i++) {
                  res = rop(res, values_ptr[i]);
                }
                new_values_ptr[row_map_ptr[h]] = res;
              }
            }
        });
                          });

  return at::native::_sparse_csr_tensor_unsafe(new_crow_indices, new_col_indices, new_values,
                                               {sparse.size(0), 1},
                                               new_values.scalar_type(),
                                               sparse.layout(),
                                               new_values.device());
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_dim01_cpu_template(const Tensor& sparse, ReductionOp rop) {

  auto ioptions = sparse.col_indices().options();
  Tensor values = sparse.values();
  auto numel = values.numel();
  auto nnz = std::min<int64_t>(1, numel);

  /* TODO: we can likely do about 3x better than parallel_reduce:

In [2]: t=torch.randn(5000, 5000).to_sparse_csr()

In [3]: %timeit torch._sparse_csr_sum(t, dim=(0, 1), keepdim=True)
3.39 ms ± 898 ns per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [4]: %timeit torch.sum(t.values())
1.07 ms ± 291 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
  */
  scalar_t* values_ptr = values.data_ptr<scalar_t>();
  scalar_t value = at::parallel_reduce(
                                       0,
                                       numel,
                                       internal::GRAIN_SIZE,
                                       rop.identity(),
                                       [&](int64_t i_start, int64_t i_end, scalar_t identity) {
                                         scalar_t res = identity;
                                         for (int64_t i=i_start; i<i_end; i++) {
                                           scalar_t val = values_ptr[i];
                                           res = rop(res, val);
                                         }
                                         return res;
                                       }, rop
                                       );

  Tensor new_col_indices = at::zeros({nnz}, ioptions);
  Tensor new_crow_indices = at::tensor(ArrayRef<int64_t>{0, nnz}, ioptions);
  Tensor new_values;
  if (numel > 0) {
    new_values = at::empty({1}, values.options());
    new_values.fill_(value);
  } else {
    new_values = at::empty({}, values.options());
  }
  return at::native::_sparse_csr_tensor_unsafe(new_crow_indices, new_col_indices, new_values,
                                               {1, std::min<int64_t>(1, sparse.size(1))},
                                               new_values.scalar_type(),
                                               sparse.layout(),
                                               new_values.device());
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_cpu_template(const Tensor& sparse, std::vector<int64_t> dims, ReductionOp rop) {
  if (dims.size() == 1) {
    if (dims[0] == 0) {
      return reduce_sparse_csr_dim0_cpu_template<scalar_t>(sparse, rop);
    } else {
      TORCH_INTERNAL_ASSERT(dims[0] == 1);
      return reduce_sparse_csr_dim1_cpu_template<scalar_t>(sparse, rop);
    }
  } else if (dims.size() == 2) {
    TORCH_INTERNAL_ASSERT(((dims[0] == 0 && dims[1] == 1) || (dims[0] == 1 && dims[1] == 0)));
    return reduce_sparse_csr_dim01_cpu_template<scalar_t>(sparse, rop);
  }
  TORCH_INTERNAL_ASSERT(dims.size() == 0);
  // effective after gh-29137 has been resolved
  return sparse.clone();
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_cpu_template(const Tensor& sparse, IntArrayRef dims_to_sum, bool keepdim, ReductionOp rop) {
  TORCH_INTERNAL_ASSERT(sparse.is_sparse_csr());
  TORCH_CHECK(keepdim, "reduction operations on CSR tensors with keepdim=False is unsupported");
  TORCH_INTERNAL_ASSERT(sparse.device() == kCPU);

  const int64_t input_dim = sparse.dim();
  TORCH_INTERNAL_ASSERT(input_dim == 2);
  auto dims = dims_to_sum.vec();
  maybe_wrap_dims(dims, input_dim);
  if (dims.size() == 0) {
    // after gh-29137 is resolved, delete this if-block
    dims.emplace_back(0);
    dims.emplace_back(1);
  }
  return reduce_sparse_csr_cpu_template<scalar_t>(sparse, dims, rop);
}

template <typename scalar_t>
struct ReductionAddOp {
  inline scalar_t operator()(const scalar_t& a, const scalar_t& b) const {
    return a + b;
  }
  inline scalar_t identity() const { return 0; }
};

template <typename scalar_t>
struct ReductionMulOp {
  inline scalar_t operator()(const scalar_t& a, const scalar_t& b) const {
    return a * b;
  }
  inline scalar_t identity() const { return 1; }
};

}  // namespace

Tensor _sparse_csr_sum_cpu(const Tensor& input, IntArrayRef dims_to_sum, bool keepdim, c10::optional<ScalarType> dtype) {
  ScalarType dtype_ = dtype.value_or(input.scalar_type());
  Tensor input_ = input.to(dtype_);
  Tensor result;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
    kHalf, kBFloat16, input_.scalar_type(), "_sparse_csr_sum_cpu",
    [&] {
      result = reduce_sparse_csr_cpu_template<scalar_t>(input_, dims_to_sum, keepdim, ReductionAddOp<scalar_t>());
    });
  return result;
}

Tensor _sparse_csr_prod_cpu(const Tensor& input, IntArrayRef dims_to_reduce, bool keepdim, c10::optional<ScalarType> dtype) {
  ScalarType dtype_ = dtype.value_or(input.scalar_type());
  Tensor input_ = input.to(dtype_);
  Tensor result;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
    kHalf, kBFloat16, input_.scalar_type(), "_sparse_csr_prod_cpu",
    [&] {
      result = reduce_sparse_csr_cpu_template<scalar_t>(input_, dims_to_reduce, keepdim, ReductionMulOp<scalar_t>());
    });
  return result;
}

} // namespace native
} // namespace at
