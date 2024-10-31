#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/mkl/Sparse.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/TensorConversions.h>
#include <ATen/native/mkl/SparseBlasImpl.h>
#include <ATen/native/sparse/SparseBlasImpl.h>
#include <ATen/native/sparse/SparseCsrTensorMath.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>
#include <ATen/AccumulateType.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#else
#include <ATen/ops/_conj_physical_native.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo_native.h>
#include <ATen/ops/_sparse_bsr_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_compressed_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csr_prod_native.h>
#include <ATen/ops/_sparse_csr_sum_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_mm_reduce_impl_backward_native.h>
#include <ATen/ops/_sparse_mm_reduce_impl_native.h>
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
#include <ATen/ops/deg2rad.h>
#include <ATen/ops/deg2rad_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/erf.h>
#include <ATen/ops/erf_native.h>
#include <ATen/ops/erfinv.h>
#include <ATen/ops/erfinv_native.h>
#include <ATen/ops/expm1.h>
#include <ATen/ops/expm1_native.h>
#include <ATen/ops/fill_native.h>
#include <ATen/ops/floor.h>
#include <ATen/ops/floor_native.h>
#include <ATen/ops/frac.h>
#include <ATen/ops/frac_native.h>
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
#include <ATen/ops/mul.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/neg_native.h>
#include <ATen/ops/normal_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/rad2deg.h>
#include <ATen/ops/rad2deg_native.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/relu_native.h>
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
#include <ATen/ops/sparse_mask.h>
#include <ATen/ops/sparse_mask_native.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/sqrt_native.h>
#include <ATen/ops/tan.h>
#include <ATen/ops/tan_native.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/tanh_native.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/threshold_backward.h>
#include <ATen/ops/threshold_backward_native.h>
#include <ATen/ops/trunc.h>
#include <ATen/ops/trunc_native.h>
#include <ATen/ops/zero_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

#include <algorithm>

namespace at {
namespace meta {

TORCH_META_FUNC(_convert_indices_from_coo_to_csr)
(const Tensor& self, const int64_t size, const bool out_int32) {
  TORCH_CHECK(self.dim() <= 1, "Input is supposed to be a vector, but got ",
              self.dim(), " dimensional tensor.");
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options =
      TensorOptions().device(self.options().device()).dtype(scalar_type);
  set_output_raw_strided(0, size + 1, {}, options);
}

TORCH_META_FUNC(_convert_indices_from_csr_to_coo)
(const Tensor& crow_indices,
 const Tensor& col_indices,
 const bool out_int32,
 const bool transpose) {
  TORCH_CHECK(
    crow_indices.dim() == col_indices.dim(), "crow_indices and col_indices are supposed to have"
    " the same dimensionality, but got ", crow_indices.dim(), " and ",
    crow_indices.dim(), " dimensional tensors, respectively.");
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = crow_indices.options().dtype(scalar_type);
  set_output_raw_strided(0, {col_indices.dim() + 1, col_indices.numel()}, {}, options, {});
}

} // namespace meta

namespace {

template <typename F>
Tensor& unary_op_out(F op_out, const Tensor& self, Tensor& result) {
  TORCH_INTERNAL_ASSERT(self.is_sparse_csr());
  TORCH_INTERNAL_ASSERT(result.is_sparse_csr());

  if (!result.is_same(self)) {
    // For the case of (0x0) result tensor, manually resize `result` tensor
    // to the size of `self` tensor
    if (result.numel() == 0) {
      at::native::resize_as_sparse_compressed_(result, self);
    }
    // copy_sparse_compressed_ internally checks the sizes of result and self tensors
    // Hence no external size check required
    at::native::copy_sparse_compressed_(result, self);
  }

  auto self_values = self.values();
  auto result_values = result.values();

  op_out(self_values, result_values);
  return result;
}

template <typename F, typename... Args>
Tensor& unary_op_inplace(Tensor& self, const F& op_inplace, Args&&... args) {
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(self.layout(), "unary_op_inplace", [](){});

  auto self_values = self.values();
  (self_values.*op_inplace)(std::forward<Args>(args)...);
  return self;
}

} // end anonymous namespace

namespace native {

using namespace at::sparse_csr;
// certain utility functions are usable from sparse COO.
using namespace at::sparse;

Tensor& mul_out_sparse_csr(const Tensor& t_, const Tensor& src_, Tensor& r) {
  // // TODO: Use a specialized CSR kernel for performance if needed
  if (t_.is_sparse_csr() && src_.layout() == kStrided) {
    return mul_out_sparse_csr(t_, src_.sparse_mask(t_), r);
  }
  if (t_.layout() == kStrided && src_.is_sparse_csr()) {
    return mul_out_sparse_csr(t_.sparse_mask(src_), src_, r);
  }
  TORCH_CHECK(r.is_sparse_csr(), "Expected result Tensor to be of format CSR");
  Tensor t = t_.to_sparse();
  Tensor src = src_.to_sparse();
  Tensor tmp_result = t.mul(src);
  auto r_sparse_csr = tmp_result.to_sparse_csr();
  r.resize_as_sparse_(r_sparse_csr);
  r.copy_(r_sparse_csr);
  return r;
}

template <typename op_t>
Tensor intersection_binary_op_with_wrapped_scalar(const Tensor& sparse, const Tensor& scalar, const op_t& op) {
  // NOTE: intersection_binary_op_with_wrapped_scalar assumes scalar.numel() == 1.
  const auto result_values = op(sparse.values(), scalar.squeeze()).to(at::result_type(sparse, scalar));
  const auto result_sizes = infer_size(sparse.sizes(), scalar.sizes());
  auto [compressed_indices, plain_indices] = getCompressedPlainIndices(sparse);
  return at::_sparse_compressed_tensor_unsafe(
      compressed_indices.clone(),
      plain_indices.clone(),
      result_values,
      result_sizes,
      sparse.options().dtype(result_values.scalar_type()));
}

template <typename op_t>
Tensor& intersection_binary_op_with_wrapped_scalar_(Tensor& sparse, const Tensor& scalar, const string& op_name, const op_t& op) {
  // NOTE: intersection_binary_op_with_wrapped_scalar_ assumes scalar.numel() == 1.
  const auto broadcasted_shape = infer_size(sparse.sizes(), scalar.sizes());
  if (sparse.sizes() != broadcasted_shape) {
    TORCH_CHECK(false, op_name, "(): output with shape ", sparse.sizes(), " does not match ",
        "the broadcast shape ", broadcasted_shape);
  }
  auto values = sparse.values();
  // Safe to use squeeze here, we already know that scalar safely broadcasts.
  op(values, scalar.squeeze());
  return sparse;
}

Tensor mul_sparse_csr(const Tensor& self, const Tensor& other) {
  // Check if either of the arguments is a wrapped Scalar
  if (self.layout() == kStrided && self.dim() == 0) {
    return intersection_binary_op_with_wrapped_scalar(other, self, [](const Tensor& a, const Tensor& b) -> Tensor {
        return a.mul(b);
    });
  }
  if (other.layout() == kStrided && other.dim() == 0) {
    return intersection_binary_op_with_wrapped_scalar(self, other, [](const Tensor& a, const Tensor& b) -> Tensor {
        return a.mul(b);
    });
  }

  if (self.is_sparse_csr() && other.layout() == kStrided) {
    return mul_sparse_csr(self, other.sparse_mask(self));
  }
  if (self.layout() == kStrided && other.is_sparse_csr()) {
    return mul_sparse_csr(self.sparse_mask(other), other);
  }

  auto commonDtype = at::result_type(self, other);
  auto result_options = self.options().dtype(commonDtype);
  // CSR is 2d!
  Tensor result = at::empty({0, 0}, result_options);
  return at::mul_out(result, self, other); // redispatch!
}

Tensor& mul_sparse_csr_(Tensor& self, const Tensor& other) {
  if (other.layout() == kStrided && other.dim() == 0) {
    return intersection_binary_op_with_wrapped_scalar_(self, other, "mul_", [](Tensor& a, const Tensor& b) -> Tensor& {
        return a.mul_(b);
    });
  }
  return at::mul_out(self, self, other); // redispatch!
}


namespace {

template <typename F>
inline Tensor get_result_tensor_for_unary_op(F op, const Tensor& input) {
  auto values = input.values();

  // To handle type promotion for inputs to unary ops,
  // we first get the result from the underlined op, and use the result
  // to create a sparse compressed tensor, which is used as the input to the out=
  // variant
  auto result_values = op(values);

  auto compressed_indices = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(input.layout(),
                                                                      "get_result_tensor_for_unary_op",
                                                                      [&]{ return input.crow_indices(); },
                                                                      [&]{ return input.ccol_indices(); });
  auto plain_indices = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(input.layout(),
                                                                 "get_result_tensor_for_unary_op",
                                                                 [&]{ return input.col_indices(); },
                                                                 [&]{ return input.row_indices(); });

  auto result = at::_sparse_compressed_tensor_unsafe(
      compressed_indices.clone(),
      plain_indices.clone(),
      result_values,
      input.sizes(),
      input.options().dtype(result_values.scalar_type()));

  return result;
}
} // namespace

Tensor& normal_sparse_csr_(
    Tensor& self,
    double mean,
    double std,
    std::optional<Generator> gen) {
  return unary_op_inplace(self, &Tensor::normal_, mean, std, gen);
}

Tensor& fill_sparse_csr_(Tensor& self, const Scalar& value) {
  return unary_op_inplace(self, &TensorBase::fill_, value);
}

Tensor sparse_mask_sparse_compressed(
    const Tensor& self,
    const Tensor& mask) {
  TORCH_CHECK(at::sparse_csr::is_sparse_compressed(mask),
              "sparse_mask_sparse_compressed expects mask to have sparse compressed layout, got ", mask.layout());
  TORCH_CHECK(
      mask.sizes().equals(self.sizes()),
      "sparse_mask(): operands have incompatible sizes; self has size ",
      self.sizes(),
      " but mask has size ",
      mask.sizes());

  if (self.is_same(mask)) {
    return self;
  }

  if (!mask.numel() || !mask._nnz()) {
    return mask.clone().to(self.device(), self.scalar_type());
  }

  if (self.layout() == kStrided) {
    auto [compressed_indices, plain_indices] = at::sparse_csr::getCompressedPlainIndices(mask);
    auto mask_values = mask.values();
    auto dense_mask = at::_sparse_compressed_tensor_unsafe(
        compressed_indices,
        plain_indices,
        at::ones({1}, self.options().dtype(kBool)).expand_as(mask_values),
        self.sizes(),
        self.options().dtype(kBool).layout(mask.layout())).to_dense();
    return AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(
        mask.layout(), "sparse_mask_sparse_compressed",
        [&] {
          return at::native::dense_to_sparse_with_mask(self, dense_mask, mask.layout(), {}, mask.dense_dim());
        },
        [&] {
          auto blocksize = at::sparse_csr::getBlockSize(mask);
          return at::native::dense_to_sparse_with_mask(self, dense_mask, mask.layout(), blocksize, mask.dense_dim());
        });
  } else if (self.layout() == mask.layout()) {
    // TODO: keeping this for BC but the method used here may lead to
    // incorrect indices.
    return self.mul(at::ones_like(mask)).to(self.scalar_type());
  } else {
    // TODO: keeping this for BC but the method used here cannot
    // support batch dimensions because sparse COO tensors are batch
    // dimension ignorant.
    return AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(
        mask.layout(), "sparse_mask_sparse_compressed",
        [&] {
          return self.sparse_mask(mask.to_sparse()).to_sparse(mask.layout());
        },
        [&] {
          auto blocksize = at::sparse_csr::getBlockSize(mask);
          return self.sparse_mask(mask.to_sparse()).to_sparse(mask.layout(), blocksize);
        });
  }
}

Tensor mul_scalar_sparse_csr(const Tensor& self, const Scalar& other) {
  auto result_values = self.values().mul(other);
  return at::native::_sparse_csr_tensor_unsafe(
      self.crow_indices().clone(),
      self.col_indices().clone(),
      result_values,
      self.sizes(),
      result_values.scalar_type(),
      self.layout(),
      result_values.device());
}

Tensor& zero_sparse_csr_(Tensor& self) {
  /*
    csr.zero_() resets nnz to 0.

    If the original sparsity pattern needs to be preserved, use
    `csr.values().zero_()` instead.

    The above behavior also implies that torch.zeros_like(csr) returns
    a new tensor with nnz == 0. If one needs a zeros_like semantics
    where the result has the same sparsity pattern as input, then use
    `result = csr.clone(); result.values.zero_();`
  */
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(self.layout(), "zero_sparse_csr_", [](){});
  get_sparse_csr_impl(self)->resize_and_clear_(self.sparse_dim(), self.dense_dim(), self.sizes());
  return self;
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

// Exhaustive list of the unary ufuncs supported by sparse compressed
CREATE_UNARY_UFUNC(abs);
CREATE_UNARY_UFUNC(asin);
CREATE_UNARY_UFUNC(asinh);
CREATE_UNARY_UFUNC(atan);
CREATE_UNARY_UFUNC(atanh);
CREATE_UNARY_UFUNC(ceil);
CREATE_UNARY_UFUNC(deg2rad);
CREATE_UNARY_UFUNC(erf);
CREATE_UNARY_UFUNC(erfinv);
CREATE_UNARY_UFUNC(expm1);
CREATE_UNARY_UFUNC(floor);
CREATE_UNARY_UFUNC(frac);
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

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-function")
static CREATE_UNARY_UFUNC(relu);
C10_DIAGNOSTIC_POP()

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

Tensor threshold_backward_sparse_compressed(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold) {
  return get_result_tensor_for_unary_op(
      [&](const Tensor& t) {
        return at::threshold_backward(t, self.values(), threshold);
      },
      grad_output);
}

Tensor& threshold_backward_sparse_compressed_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold,
    Tensor& grad_input) {
  return unary_op_out(
      [&](const Tensor& t, Tensor& out) {
        return at::threshold_backward_outf(t, self.values(), threshold, out);
      },
      grad_output,
      grad_input);
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
Tensor& addmm_out_sparse_compressed_cpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
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
    // If result gets resized and is sparse compressed,
    // it's compressed_indices tensor will contain junk values
    // so the whole tensor is not a valid compressed tensor.
    // To combat that, result needs to get zeroed out.
    if (at::sparse_csr::is_sparse_compressed(result)) {
      result.zero_();
    }
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
  // The custom impl addmm_out_sparse_csr_native_cpu only supports CSR @
  // strided -> strided
  if (mat1.layout() == kStrided) {
    if (mat2.layout() == kSparseCsr) {
      if (result.layout() == kStrided) {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
            result.scalar_type(), "addmm_sparse_dense", [&] {
              addmm_out_sparse_csr_native_cpu<scalar_t>(
                  mat2.transpose(-2, -1).to_sparse_csr(),
                  mat1.transpose(-2, -1),
                  result.transpose(-2, -1),
                  alpha,
                  beta);
            });
        return result;
      }
    }
    if (mat2.layout() == kSparseCsc) {
      if (result.layout() == kStrided) {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
            result.scalar_type(), "addmm_sparse_dense", [&] {
              addmm_out_sparse_csr_native_cpu<scalar_t>(
                  mat2.transpose(-2, -1),
                  mat1.transpose(-2, -1),
                  result.transpose(-2, -1),
                  alpha,
                  beta);
            });
        return result;
      }
    }
  } else if (mat1.layout() == kSparseCsr) {
    if (mat2.layout() == kStrided) {
      if (result.layout() == kStrided) {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
            result.scalar_type(), "addmm_sparse_dense", [&] {
              addmm_out_sparse_csr_native_cpu<scalar_t>(
                  mat1, mat2, result, alpha, beta);
            });
        return result;
      }
    }
  } else if (mat1.layout() == kSparseCsc) {
    if (mat2.layout() == kStrided) {
      if (result.layout() == kStrided) {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
            result.scalar_type(), "addmm_sparse_dense", [&] {
              addmm_out_sparse_csr_native_cpu<scalar_t>(
                  mat1.to_sparse_csr(), mat2, result, alpha, beta);
            });
        return result;
      }
    }
  }
  TORCH_CHECK(
      false,
      "addmm: computation on CPU is not implemented for ",
      result.layout(),
      " + ",
      mat1.layout(),
      " @ ",
      mat2.layout(),
      " without MKL. PyTorch built with MKL has better support for addmm with sparse CPU tensors.");
#else
  sparse::impl::mkl::addmm_out_sparse_csr(mat1, mat2, beta, alpha, result);
#endif
  return result;
}

Tensor addmm_sparse_compressed_dense(
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
  auto zero = at::zeros_like(result);
  return at::addmm_out(result, zero, mat1, mat2, 0.0, 1.0);
}

Tensor _sparse_csr_mm(const Tensor& mat1, const Tensor& mat2) {
  if (mat1.is_sparse_csr() && mat2.is_sparse_csr()) {
    // Return sparse
    return at::addmm(
        at::zeros({mat1.size(0), mat2.size(1)}, mat2.options()),
        mat1,
        mat2,
        0.0,
        1.0);
  }
  if ((mat1.layout() == kSparseCsc || mat1.layout() == kSparseCsr) &&
      (mat2.layout() == kSparseCsc || mat2.layout() == kSparseCsr)) {
    // TODO: Expensive conversion to CSR. Should add native support for CSC.
    // Covers CSC @ CSR
    // Covers CSR @ CSC
    // Covers CSC @ CSC
    return _sparse_csr_mm(mat1.to_sparse_csr(), mat2.to_sparse_csr());
  }
  if (mat1.layout() == kSparseCsc && mat2.layout() == c10::kStrided) {
    // TODO: This is a costly conversion. We should have
    // native support for CSC.
    return _sparse_csr_mm(mat1.to_sparse_csr(), mat2);
  }
  // Default to taking options from mat1
  auto result_options = mat1.options();
  if (mat2.layout() == kStrided) {
    // if either  arg is strided we return strided, so update the options if
    // mat2 is strided.
    result_options = result_options.layout(kStrided);
  }
  return at::addmm(
      at::zeros({mat1.size(0), mat2.size(1)}, result_options),
      mat1,
      mat2,
      0.0,
      1.0);
}

// Functions for element-wise addition.
Tensor add_sparse_csr(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  auto commonDtype = at::result_type(self, other);
  alpha_check(commonDtype, alpha);
  Tensor result;
  if (self.layout() != kStrided && other.layout() == kStrided) {
    // add(sparse, dense) -> dense
    result = at::empty_like(
        other,
        other.options()
            .dtype(commonDtype)
            .memory_format(at::MemoryFormat::Contiguous));
  } else {
    // add(dense, sparse) -> dense AND add(sparse, sparse) -> sparse
    result = at::empty_like(
        self,
        self.options()
            .dtype(commonDtype)
            .memory_format(at::MemoryFormat::Contiguous));
  }
  return at::add_out(result, self, other, alpha); // redispatch!
}

Tensor& add_sparse_csr_(
    Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  return at::add_out(self, self, other, alpha); // redispatch!
}

static void add_out_dense_sparse_compressed_cpu(
    const Tensor& out,
    const Tensor& dense,
    const SparseCsrTensor& src,
    const Scalar& alpha) {
  TORCH_INTERNAL_ASSERT(dense.layout() == kStrided);
  TORCH_INTERNAL_ASSERT(
      src.layout() == kSparseCsr || src.layout() == kSparseCsc);
  TORCH_INTERNAL_ASSERT(dense.device() == kCPU || dense.device() == kMeta);

  TORCH_CHECK(
      out.is_contiguous(),
      "out argument must be contiguous, but got: ",
      out.suggest_memory_format());
  TORCH_CHECK(
      out.device() == dense.device(),
      "add: expected 'out' to match dense tensor, but got tensor on device: ",
      out.device());
  TORCH_CHECK(
      src.device() == dense.device(),
      "add: expected 'src' to match dense tensor, but got tensor on device: ",
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

  TORCH_INTERNAL_ASSERT(dense.device() == kCPU);

  auto valuesBuffer = src_values.to(commonDtype).reshape({-1, src_values.size(-1)});
  resultBuffer = resultBuffer.view({-1, out.size(-2), out.size(-1)});
  Tensor src_compressed_indices;
  Tensor src_plain_indices;
  std::tie(src_compressed_indices, src_plain_indices) =
      at::sparse_csr::getCompressedPlainIndices(src);
  src_compressed_indices =
      src_compressed_indices.reshape({-1, src_compressed_indices.size(-1)});
  src_plain_indices =
      src_plain_indices.reshape({-1, src_plain_indices.size(-1)});
  auto src_layout = src.layout();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kComplexHalf,
      kHalf,
      kBool,
      kBFloat16,
      commonDtype,
      "add_out_op2_sparse_csr",
      [&valuesBuffer,
       &resultBuffer,
       &alpha,
       &src_compressed_indices,
       &src_plain_indices,
       &src_layout]() {
        AT_DISPATCH_INDEX_TYPES(
            src_compressed_indices.scalar_type(),
            "csr_add_out_crow_indices",
            [&valuesBuffer,
             &resultBuffer,
             &alpha,
             &src_compressed_indices,
             &src_plain_indices,
             &src_layout]() {
              auto batch_count =
                  resultBuffer.dim() > 2 ? resultBuffer.size(-3) : 1;
              auto values_accessor = valuesBuffer.accessor<scalar_t, 2>();
              scalar_t* out_ptr = resultBuffer.data_ptr<scalar_t>();
              scalar_t cast_value = alpha.to<scalar_t>();

              auto compressed_indices_accessor =
                  src_compressed_indices.accessor<index_t, 2>();
              auto plain_indices_accessor =
                  src_plain_indices.accessor<index_t, 2>();
              auto out_strides = resultBuffer.strides();
              auto const out_stride_batch = out_strides[0];
              auto const out_stride_compressed =
                  AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
                      src_layout,
                      "add_out_dense_sparse_compressed_cpu",
                      [&out_strides] { return out_strides[1]; },
                      [&out_strides] { return out_strides[2]; });
              auto const out_stride_plain =
                  AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
                      src_layout,
                      "add_out_dense_sparse_compressed_cpu",
                      [&out_strides] { return out_strides[2]; },
                      [&out_strides] { return out_strides[1]; });

              for (const auto batch_idx : c10::irange(batch_count)) {
                for (const auto i_compressed :
                     c10::irange(src_compressed_indices.size(-1) - 1)) {
                  index_t start_index =
                      compressed_indices_accessor[batch_idx][i_compressed];
                  index_t end_index =
                      compressed_indices_accessor[batch_idx][i_compressed + 1];
                  for (const auto i : c10::irange(start_index, end_index)) {
                    auto i_plain = plain_indices_accessor[batch_idx][i];
                    auto index = batch_idx * out_stride_batch +
                        i_compressed * out_stride_compressed +
                        i_plain * out_stride_plain;
                    out_ptr[index] +=
                        cast_value * values_accessor[batch_idx][i];
                  }
                }
              }
            });
      });
  if (out.scalar_type() != commonDtype) {
    out.copy_(resultBuffer);
  }
}

Tensor& add_out_sparse_compressed_cpu(
    const Tensor& self,
    const SparseCsrTensor& other,
    const Scalar& alpha,
    SparseCsrTensor& out) {
  if (self.layout() == kStrided) {
    add_out_dense_sparse_compressed_cpu(out, self, other, alpha);
  } else if (other.layout() == kStrided) {
    add_out_dense_sparse_compressed_cpu(out, other, self, alpha);
  } else {
    TORCH_CHECK(
        self.sizes().equals(other.sizes()),
        "torch.add: Expected input tensors to have the same shape, but got tensor `self` with shape ",
        self.sizes(),
        " and tensor `other` with shape ",
        other.sizes());

    if (only_sparse_compressed_add_trivial_cases(self, other, alpha, out)) {
      return out;
    }

    at::native::resize_as_sparse_compressed_(out, self);
    sparse::impl::cpu::add_out_sparse_csr(self, other, alpha, out);
  }
  return out;
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

Tensor _sparse_csr_..._cpu(const Tensor& input, IntArrayRef dims_to_sum, bool keepdim, std::optional<ScalarType> dtype) {
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

  /*
    Calling at::_unique constitutes the main bottleneck of this
    function. However, it is still about 5x faster than using the
    invariant:
      csr.sum(dim=0) == csr.transpose(0, 1).sum(dim=1)
  */
  auto [new_col_indices, columns_map] = at::_unique(col_indices, true, true);
  auto nnz = new_col_indices.numel();

  Tensor new_crow_indices = at::empty({2}, col_indices.options());
  new_crow_indices[0] = 0;
  new_crow_indices[1] = nnz;

  // Set `is_cuda` = `true` in acc_type in CPU backend. Because the accumulate type
  // of float should be float in current scenario. In CUDA, float is the accumulate type
  // of float, while in CPU, double is the accumulate type of float.
  using acc_t = at::acc_type<scalar_t, true>;
  auto acc_buffer = at::sparse_csr::create_acc_buffer<acc_t, scalar_t>(
      values.options(), values.scalar_type(), nnz);
  Tensor new_values = std::get<0>(acc_buffer);
  Tensor new_values_acc = std::get<1>(acc_buffer);
  new_values_acc.fill_(rop.identity());

  int64_t* columns_map_ptr = columns_map.data_ptr<int64_t>();
  scalar_t* values_ptr = values.data_ptr<scalar_t>();
  acc_t* new_values_acc_ptr =
      new_values_acc.data_ptr<acc_t>();

  // There is no point in parallelizing the following for-loop
  // because about 99.3% of the computation time is spent in the
  // at::_unique call above.
  for (const auto i : c10::irange(numel)) {
    int64_t col = columns_map_ptr[i];
    scalar_t val = values_ptr[i];
    new_values_acc_ptr[col] = rop(new_values_acc_ptr[col], static_cast<acc_t>(val));
  }
  copy_from_acc_buffer(new_values, new_values_acc);

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
  Tensor row_map = at::empty({nrows}, ioptions);

  // Set `is_cuda` = `true` in acc_type in CPU backend. Because the accumulate type
  // of float should be float in current scenario. In CUDA, float is the accumulate type
  // of float, while in CPU, double is the accumulate type of float.
  using acc_t = at::acc_type<scalar_t, true>;
  auto acc_buffer = at::sparse_csr::create_acc_buffer<acc_t, scalar_t>(
      values.options(), values.scalar_type());
  Tensor new_values = std::get<0>(acc_buffer);
  Tensor new_values_acc = std::get<1>(acc_buffer);

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
    new_values_acc.resize_(nnz);

    scalar_t* values_ptr = values.data_ptr<scalar_t>();
    acc_t* new_values_acc_ptr = new_values_acc.data_ptr<acc_t>();

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
                acc_t res = static_cast<acc_t>(values_ptr[i_start]);
                for (index_t i = i_start + 1; i < i_end; i++) {
                  res = rop(res, static_cast<acc_t>(values_ptr[i]));
                }
                new_values_acc_ptr[row_map_ptr[h]] = res;
              }
            }
        });
                          });

  copy_from_acc_buffer(new_values, new_values_acc);

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

  // Set `is_cuda` = `true` in acc_type in CPU backend. Because the accumulate type
  // of float should be float in current scenario. In CUDA, float is the accumulate type
  // of float, while in CPU, double is the accumulate type of float.
  using acc_t = at::acc_type<scalar_t, true>;
  scalar_t* values_ptr = values.data_ptr<scalar_t>();
  acc_t value = at::parallel_reduce(
                                       0,
                                       numel,
                                       internal::GRAIN_SIZE,
                                       rop.identity(),
                                       [&](int64_t i_start, int64_t i_end, scalar_t identity) {
                                         acc_t res = acc_t(identity);
                                         for (int64_t i=i_start; i<i_end; i++) {
                                           acc_t val = acc_t(values_ptr[i]);
                                           res = rop(res, val);
                                         }
                                         return res;
                                       }, rop
                                       );

  Tensor new_col_indices = at::zeros({nnz}, ioptions);
  Tensor new_crow_indices = at::tensor(ArrayRef<int64_t>{0, nnz}, ioptions);
  Tensor new_values;
  auto result_dtype = at::isIntegralType(values.scalar_type(), /*includeBool=*/true) ? ScalarType::Long : values.scalar_type();
  if (numel > 0) {
    new_values = at::empty({1}, values.options().dtype(result_dtype));
    new_values.fill_(value);
  } else {
    new_values = at::empty({}, values.options().dtype(result_dtype));
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
  TORCH_INTERNAL_ASSERT(dims.empty());
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
  if (dims.empty()) {
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

Tensor _sparse_csr_sum_cpu(const Tensor& input, IntArrayRef dims_to_sum, bool keepdim, std::optional<ScalarType> dtype) {
  ScalarType dtype_ = dtype.value_or(input.scalar_type());
  Tensor input_ = at::sparse_csr::to_type(input, dtype_);
  Tensor result;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf, kBFloat16, input_.scalar_type(), "_sparse_csr_sum_cpu", [&] {
        // Set `is_cuda` = `true` in acc_type in CPU backend. Because the accumulate type
        // of float should be float in current scenario. In CUDA, float is the accumulate type
        // of float, while in CPU, double is the accumulate type of float.
        using acc_t = at::acc_type<scalar_t, true>;
        result = reduce_sparse_csr_cpu_template<scalar_t>(
            input_, dims_to_sum, keepdim, ReductionAddOp<acc_t>());
      });
  return result;
}

Tensor _sparse_csr_prod_cpu(const Tensor& input, IntArrayRef dims_to_reduce, bool keepdim, std::optional<ScalarType> dtype) {
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

std::tuple<Tensor, Tensor> _sparse_mm_reduce_impl_sparse_csr_cpu(
    const Tensor& self,
    const Tensor& other,
    const c10::string_view reduce) {

  auto layout = self.layout();
  TORCH_CHECK(layout == kSparseCsr,
      "sparse_mm_reduce: expect self to be SparseCsr, got ", layout);
  TORCH_CHECK(self.dense_dim() == 0,
      "sparse_mm_reduce: expected non-hybrid self tensor.");
  TORCH_CHECK(self.dim() == 2,
      "sparse_mm_reduce: expected self to be a 2-D tensor, got ", self.dim(), "-D tensor.");

  sparse::impl::check_sparse_mm_reduce_impl_inputs</*train*/false>(
      self, Tensor(), other);

  auto op = get_reduction_enum(reduce);
  TORCH_CHECK(op != ReductionType::PROD, "sparse_mm_reduce: reduce type of prod has not been enabled.")

  auto crow = self.crow_indices();
  auto col = self.col_indices();
  auto val = self.values();

  // init output to be all zeros, for `rows` that has no nonzero elements,
  // the corresponding rows in the output will be zero.
  auto out = at::zeros({self.size(0), other.size(1)}, other.options());
  auto arg_out = at::empty({0}, col.options());

  int64_t nnz = self._nnz();
  if (nnz == 0) {
    return std::make_tuple(out, arg_out);
  }

  // only need to calculate the out args
  // for reduce type "amax" and "amin" for training
  bool need_arg_out = at::GradMode::is_enabled()
      && (self.requires_grad() || other.requires_grad())
      && (op == ReductionType::MAX || op == ReductionType::MIN);

  if (!need_arg_out) {
    spmm_reduce_stub(kCPU, out, crow, col, val, other, op);
  } else {
    // allocate memory and init with invalid index
    arg_out.resize_(out.sizes());
    arg_out.fill_(nnz);
    spmm_reduce_arg_stub(kCPU, out, arg_out, crow, col, val, other, op);
  }

  return std::make_tuple(std::move(out), std::move(arg_out));
}

std::tuple<Tensor, Tensor> _sparse_mm_reduce_impl_backward_sparse_csr_cpu(
    const Tensor& self,
    const Tensor& grad_out,
    const Tensor& other,
    const c10::string_view reduce,
    const Tensor& arg_out,
    std::array<bool, 2> output_mask) {

  auto layout = self.layout();
  TORCH_CHECK(layout == kSparseCsr,
      "sparse_mm_reduce: expect self to be SparseCsr, got ", layout);

  sparse::impl::check_sparse_mm_reduce_impl_inputs</*train*/true>(
      self, grad_out, other);

  auto op = get_reduction_enum(reduce);

  auto crow = self.crow_indices();
  auto col = self.col_indices();
  auto val = self.values();

  // `row`: row indices of COO format
  // `ccol`: ccol indices of CSC format (with permute)
  // `permute`: permute pattern from CSR to CSC
  //
  // TODO: optimize the following section,
  // currently `argsort` is sequential.
  Tensor row, ccol, permute;
  {
    bool out_int32 = crow.scalar_type() == ScalarType::Int;
    Tensor coo_indices = at::_convert_indices_from_csr_to_coo(
        crow,
        col,
        out_int32,
        /*transpose*/false);
    row = coo_indices.select(0, 0);

    // calculate the global index for CSC
    // and get the conversion permute pattern
    Tensor index = col.mul(self.size(0)).add_(row);
    permute = index.argsort();

    ccol = at::_convert_indices_from_coo_to_csr(
        /*column indices*/col.index_select(0, permute),
        /*column count*/self.size(1),
        out_int32);
  }

  Tensor grad_self, grad_other;
  if (output_mask[0]) {
    // grad_input has the same indices and nnz with input
    grad_self = at::empty_like(self);
    grad_self.values().zero_();
    if (op == ReductionType::MAX || op == ReductionType::MIN) {
      spmm_reduce_backward_input_arg_stub(kCPU, grad_self, grad_out, col, other, arg_out, op);
    } else {
      spmm_reduce_backward_input_stub(kCPU, grad_self, grad_out, crow, col, other, row, op);
    }
  }
  if (output_mask[1]) {
    grad_other = at::zeros(other.sizes(), other.options());
    if (op == ReductionType::MAX || op == ReductionType::MIN) {
      spmm_reduce_backward_other_arg_stub(kCPU, grad_other, grad_out, col, val, arg_out, op);
    } else {
      spmm_reduce_backward_other_stub(kCPU, grad_other, grad_out, crow, val, row, ccol, permute, op);
    }
  }

  return std::make_tuple(std::move(grad_self), std::move(grad_other));
}

DEFINE_DISPATCH(spmm_reduce_stub);
DEFINE_DISPATCH(spmm_reduce_arg_stub);
DEFINE_DISPATCH(spmm_reduce_backward_input_stub);
DEFINE_DISPATCH(spmm_reduce_backward_input_arg_stub);
DEFINE_DISPATCH(spmm_reduce_backward_other_stub);
DEFINE_DISPATCH(spmm_reduce_backward_other_arg_stub);

} // namespace native
} // namespace at
