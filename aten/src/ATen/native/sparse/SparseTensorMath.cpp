#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/SparseTensorMath.h>

#include <c10/util/irange.h>
#include <c10/util/MaybeOwned.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Copy.h>
#include <ATen/native/CPUBlas.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_addmm.h>
#include <ATen/ops/_sparse_addmm_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/_sparse_mm_native.h>
#include <ATen/ops/_sparse_sum.h>
#include <ATen/ops/_sparse_sum_backward_native.h>
#include <ATen/ops/_sparse_sum_native.h>
#include <ATen/ops/add.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/any.h>
#include <ATen/ops/any_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/conj_physical.h>
#include <ATen/ops/conj_physical_native.h>
#include <ATen/ops/copy_sparse_to_sparse.h>
#include <ATen/ops/div.h>
#include <ATen/ops/div_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/floor_divide.h>
#include <ATen/ops/floor_divide_native.h>
#include <ATen/ops/hspmm_native.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/mv_native.h>
#include <ATen/ops/native_norm_native.h>
#include <ATen/ops/neg_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/pow_native.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/smm_native.h>
#include <ATen/ops/sspaddmm.h>
#include <ATen/ops/sspaddmm_native.h>
#include <ATen/ops/sub_native.h>
#include <ATen/ops/zero_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/zeros_native.h>
#endif

#include <algorithm>

namespace at { namespace native {

using namespace at::sparse;
// --------------------------------------------------------------------
// Utility functions
// --------------------------------------------------------------------

namespace {

  inline SparseTensor get_result_tensor_for_unary_op(const SparseTensor& input) {
    if (c10::isIntegralType(input.scalar_type(), /*includeBool=*/true)) {
      return at::empty_like(input, input.options().dtype(c10::get_default_dtype()));
    }
    return at::empty_like(input);
  }
}

// --------------------------------------------------------------------
// zero_(SparseTensor)
// --------------------------------------------------------------------

// hummu hummu
SparseTensor& zero_sparse_(SparseTensor& self) {
  AT_ASSERT(self.is_sparse());
  at::zeros_out(self, get_sparse_impl(self)->sizes());
  return self._coalesced_(true);
}

// NB: Don't need zeros, zeros_like, already implemented in TensorFactories

// --------------------------------------------------------------------
// mul(SparseTensor, Scalar)
// --------------------------------------------------------------------

SparseTensor& mul_out_sparse_zerodim(SparseTensor& r, const SparseTensor& t, const Tensor& value) {
  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t.is_sparse());
  AT_ASSERT(value.dim() == 0);

  // Resolve a possibly sparse COO value to a strided tensor.
  Tensor value_;
  if (value.is_sparse()) {
    if (value._nnz() == 0) {
      r.resize_as_(t);
      return r.zero_();
    }
    value_ = value.values();
  } else {
    value_ = value;
  }
  // With broadcasting in action, value_ may be a 1-D tensor as long
  // as its shape is (1,).
  AT_ASSERT(value_.numel() == 1);

  if (is_same_tensor(r, t)) {
    r._values().mul_(value_);
  } else {
    r.resize_as_(t);
    auto indices = r._indices();
    indices.resize_as_(t._indices());
    indices.copy_(t._indices());
    Tensor r_values = r._values(); // Sigh... needed because mul_out takes Tensor&
    at::mul_out(r_values, t._values(), value_);
    get_sparse_impl(r)->set_nnz_and_narrow(t._nnz());
    r._coalesced_(t.is_coalesced());
  }
  return r;
}

SparseTensor& mul_out_sparse_scalar(SparseTensor& r, const SparseTensor& t, const Scalar& value) {
  return mul_out_sparse_zerodim(r, t, wrapped_scalar_tensor(value));
}

// --------------------------------------------------------------------
// neg(SparseTensor)
// --------------------------------------------------------------------

SparseTensor& neg_out_sparse(const SparseTensor& t, SparseTensor& r) {
  TORCH_CHECK(r.is_sparse(), "Tensor should be sparse");
  TORCH_CHECK(t.is_sparse(), "Tensor should be sparse");

  // copy_sparse_ does not perform the copy if it is the same tensor
  copy_sparse_to_sparse_(r, t);
  r._values().neg_();
  return r;
}

SparseTensor neg_sparse(const SparseTensor& t) {
  SparseTensor r = at::empty_like(t);
  neg_out_sparse(t, r);
  return r;
}

SparseTensor& neg_sparse_(SparseTensor& t) {
  return neg_out_sparse(t, t);
}

// --------------------------------------------------------------------
// pow(SparseTensor, Scalar)
// --------------------------------------------------------------------

// TODO: add in-place variant

SparseTensor& pow_out_sparse_scalar(const SparseTensor& t_, const Scalar& value, SparseTensor& r) {
  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t_.is_sparse());
  TORCH_CHECK(value.toDouble() != 0, "pow: cannot raise to zeroth power on sparse tensor; it would make the result tensor dense");

  // This coalesce is why we can't easily provide an inplace variant
  SparseTensor t = t_.coalesce();

  r.resize_as_(t);
  auto indices = r._indices();
  indices.resize_as_(t._indices());
  indices.copy_(t._indices());
  Tensor r_values = r._values(); // Sigh... needed because pow_out takes Tensor&
  at::pow_out(r_values, t._values(), value);
  get_sparse_impl(r)->set_nnz_and_narrow(t._nnz());
  return r._coalesced_(t.is_coalesced());
}

SparseTensor pow_sparse_scalar(const SparseTensor& t, const Scalar& value) {
  SparseTensor r = at::empty({0}, t.options());
  pow_out_sparse_scalar(t, value, r);
  return r;
}

// --------------------------------------------------------------------
// div(SparseTensor, Scalar)
// --------------------------------------------------------------------

static SparseTensor& coalesce_(SparseTensor& tensor) {
  if (tensor.is_coalesced()) {
    return tensor;
  }

  SparseTensor coalesced = tensor.coalesce();
  tensor._values().resize_as_(coalesced._values());
  tensor._indices().resize_as_(coalesced._indices());
  tensor._values().copy_(coalesced._values());
  tensor._indices().copy_(coalesced._indices());
  tensor._coalesced_(true);
  return tensor;
}

// Note [Sparse Floor Division]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Uncoalesced sparse tensors cannot be floor divided correctly. Integer
// division is considered a special-case of floor division for purposes of
// this note.
// For example, an integer tensor with values=[3, 3] divided by 2 would produce
// values=[1, 1], which sum to 2 instead of 3 (=6/2).
// A float tensor with values=[3., 3.] floor divided by 2 would also produce
// values=[1., 1.] (after truncation), which sum to 2.f instead of 3.f.
// To perform floor division the sparse tensor must be coalesced first.

SparseTensor& div_out_sparse_zerodim(const SparseTensor& t, const Tensor& value, c10::optional<c10::string_view> rounding_mode, SparseTensor& r) {
  TORCH_CHECK(value.dim() == 0, "Sparse division requires a scalar or ",
    "zero-dim dense tensor divisor (got shape ", value.sizes(), " for divisor)");
  TORCH_CHECK(!value.is_sparse(), "Sparse division requires a scalar or ",
    "zero-dim dense tensor divisor (got a sparse divisor)");

  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t.is_sparse());

  // See note "Sparse Floor Division"
  const bool should_coalesce = rounding_mode.has_value() && !t.is_coalesced();
  if (is_same_tensor(r, t)) {
    if (should_coalesce) {
      coalesce_(r);
    }
    r._values().div_(value, rounding_mode);
  } else {
    Tensor t_tmp = t;
    if (should_coalesce) {
      t_tmp = t.coalesce();
    }
    r.resize_as_(t_tmp);
    auto indices = r._indices();
    indices.resize_as_(t_tmp._indices());
    indices.copy_(t_tmp._indices());
    Tensor r_values = r._values(); // Sigh... needed because div_out takes Tensor&
    at::div_out(r_values, t_tmp._values(), value, rounding_mode);
    get_sparse_impl(r)->set_nnz_and_narrow(t_tmp._nnz());
    r._coalesced_(t_tmp.is_coalesced());
  }
  return r;
}

SparseTensor& div_out_sparse_zerodim(const SparseTensor& t, const Tensor& value, SparseTensor& r) {
  return div_out_sparse_zerodim(t, value, /*rounding_mode=*/c10::nullopt, r);
}

Tensor div_sparse(const Tensor& self, const Tensor& value) {
  auto commonDtype = at::result_type(self, value);
  if (c10::isIntegralType(commonDtype, /*includeBool=*/true)) {
    commonDtype = typeMetaToScalarType(at::get_default_dtype());
  }
  Tensor result = at::empty({0}, self.options().dtype(commonDtype));
  return div_out_sparse_zerodim(self, value, result);
}

Tensor& div_sparse_(Tensor& self, const Tensor& value) {
  return div_out_sparse_zerodim(self, value, self);
}

SparseTensor& div_out_sparse_scalar(const SparseTensor& t, Scalar value, SparseTensor& r) {
  return div_out_sparse_zerodim(t, wrapped_scalar_tensor(value), r);
}

Tensor div_sparse(const Tensor& self, const Tensor& value, c10::optional<c10::string_view> rounding_mode) {
  auto commonDtype = at::result_type(self, value);
  if (c10::isIntegralType(commonDtype, /*includeBool=*/true) && !rounding_mode.has_value()) {
    commonDtype = typeMetaToScalarType(at::get_default_dtype());
  }
  Tensor result = at::empty({0}, self.options().dtype(commonDtype));
  return div_out_sparse_zerodim(self, value, std::move(rounding_mode), result);
}

Tensor& div_sparse_(Tensor& self, const Tensor& value, c10::optional<c10::string_view> rounding_mode) {
  return div_out_sparse_zerodim(self, value, std::move(rounding_mode), self);
}

SparseTensor& div_out_sparse_scalar(const SparseTensor& t, Scalar value, c10::optional<c10::string_view> rounding_mode, SparseTensor& r) {
  return div_out_sparse_zerodim(t, wrapped_scalar_tensor(value), std::move(rounding_mode), r);
}

// --------------------------------------------------------------------
// floor_divide(SparseTensor, Scalar)
// --------------------------------------------------------------------

SparseTensor& floor_divide_out_sparse_zerodim(const SparseTensor& dividend,
  const Tensor& divisor,
  SparseTensor& result) {
  TORCH_CHECK(divisor.dim() == 0, "Sparse floor division requires a scalar or ",
    "zero-dim dense tensor divisor (got shape ", divisor.sizes(), " for divisor)");
  TORCH_CHECK(!divisor.is_sparse(), "Sparse floor division requires a scalar or ",
    "zero-dim dense tensor divisor (got a sparse divisor)");

  AT_ASSERT(result.is_sparse());
  AT_ASSERT(dividend.is_sparse());

  // Case 1: result and dividend are the same tensor
  // Performs floor division in-place
  if (is_same_tensor(result, dividend)) {

    // See note "Sparse Floor Division"
    if (!result.is_coalesced()) {
      coalesce_(result);
    }

    result._values().floor_divide_(divisor);
    return result;
  }

  // Case 2: result and dividend are different tensors
  Tensor dividend_tmp = dividend;

  // Ensures dividend_tmp is coalesced (see note above)
  if (!dividend.is_coalesced()) {
    dividend_tmp = dividend.coalesce();
  }

  // Resizes and indexes result like dividend_tmp
  result.resize_as_(dividend_tmp);
  result._indices().resize_as_(dividend_tmp._indices());
  result._indices().copy_(dividend_tmp._indices());

  // Computes result
  Tensor result_values = result._values();
  at::floor_divide_out(result_values, dividend_tmp._values(), divisor);
  get_sparse_impl(result)->set_nnz_and_narrow(dividend_tmp._nnz());
  result._coalesced_(dividend_tmp.is_coalesced());
  return result;
}

Tensor floor_divide_sparse(const Tensor& self, const Tensor& value) {
  auto commonDtype = at::result_type(self, value);
  Tensor result = at::empty({0}, self.options().dtype(commonDtype));
  return floor_divide_out_sparse_zerodim(self, value, result);
}

Tensor& floor_divide_sparse_(Tensor& self, const Tensor& value) {
  return floor_divide_out_sparse_zerodim(self, value, self);
}

SparseTensor& floor_divide_out_sparse_scalar(SparseTensor& r, const SparseTensor& t, const Scalar& value) {
  return floor_divide_out_sparse_zerodim(t, wrapped_scalar_tensor(value), r);
}

// --------------------------------------------------------------------
// norm(SparseTensor, Scalar)
// --------------------------------------------------------------------

// Only supports floating point, FYI
Tensor norm_sparse(const SparseTensor& self, const Scalar& p) {
  AT_ASSERT(self.is_sparse());
  return norm_sparse(self, p, IntArrayRef{}, false, c10::nullopt);
}

Tensor norm_sparse(const SparseTensor& self, const optional<Scalar>& p, IntArrayRef dim, bool keepdim, optional<ScalarType> dtype) {
  AT_ASSERT(self.is_sparse());
  if (dim.size() > 0) {
    // Only full reductions are supported, so check if that is the case
    int64_t ndim = self.dim();
    bool passed_full_reduction_check = static_cast<size_t>(ndim) == dim.size();
    if (passed_full_reduction_check) {
      auto dim_ = dim.vec();
      maybe_wrap_dims(dim_, ndim);
      std::vector<bool> dims_check(ndim, false);
      // Need to check for duplicates, and fail if any are found
      for (auto dim_ind : dim_) {
        if (dims_check[dim_ind]) {
          passed_full_reduction_check = false;
          break;
        }
        dims_check[dim_ind] = true;
      }
    }
    TORCH_CHECK(passed_full_reduction_check,
      "norm_sparse currently only supports full reductions, so 'dim' must either be empty or contain all dimensions of the input");
  }
  TORCH_CHECK(keepdim == false, "norm_sparse currently does not support keepdim=True");
  TORCH_CHECK(!dtype.has_value(), "norm_sparse currently does not support 'dtype' argument");
  constexpr auto TWO = 2.0;
  auto p_ = p.value_or(TWO);
  return self.coalesce()._values().norm(p_);
}

// --------------------------------------------------------------------
// mv(SparseTensor, Tensor)
// --------------------------------------------------------------------

Tensor mv_sparse(const SparseTensor& self, const Tensor& vec)
{
  TORCH_CHECK(self.ndimension() == 2 &&
              vec.ndimension() == 1,
              "mv: two tensor dim should be 2 and 1, but got ",
              "SparseTensor Dim: ", self.ndimension(), "Tensor Dim: ", vec.ndimension());

  TORCH_CHECK(vec.size(-1) == self.size(-1),
              "mv: expected self.size(-1) == vec.size(-1)");

  auto result = self.matmul(vec.unsqueeze(-1));

  return result.squeeze(-1);
}

// --------------------------------------------------------------------
// add(SparseTensor, SparseTensor, Scalar)  [broadcasts]
// --------------------------------------------------------------------

Tensor add_sparse(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  // TODO: Why?! Can't we just flip the order here...
  TORCH_CHECK(!(self.is_sparse() && !other.is_sparse()),
              "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  auto commonDtype = at::result_type(self, other);
  alpha_check(commonDtype, alpha);
  Tensor result = at::empty({0}, self.options().dtype(commonDtype));
  return at::add_out(result, self, other, alpha);  // redispatch!
}

Tensor& add_sparse_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return at::add_out(self, self, other, alpha);  // redispatch!
}

// There's actually nothing sparse specific about these implementations

Tensor sub_sparse(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  sub_check(self, other);
  return native::add_sparse(self, other, -alpha);
}

Tensor& sub_sparse_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  sub_check(self, other);
  return native::add_sparse_(self, other, -alpha);
}

Tensor& sub_out_sparse(const Tensor& self, const Tensor& other, const Scalar& alpha, Tensor& r) {
  sub_check(self, other);
  return at::add_out(r, self, other, -alpha);  // redispatch!
}


SparseTensor& add_out_sparse_contiguous(SparseTensor& r, const SparseTensor& t, const SparseTensor& src, const Scalar& value, ScalarType commonDtype) {
    // saving those because they can be overwritten when doing in-place operations
    int64_t t_nnz = t._nnz(), s_nnz = src._nnz(), max_nnz = t_nnz + s_nnz;
    bool coalesced = t.is_coalesced() && src.is_coalesced();
    int64_t sparse_dim = src.sparse_dim();

    Tensor r_indices = at::empty({src.sparse_dim(), max_nnz}, t._indices().options());

    Tensor t_values = t._values().to(commonDtype);
    Tensor s_values = src._values().to(commonDtype);

    Tensor r_values = new_values_with_size_of(s_values, max_nnz).zero_();

    int64_t blockSize = r_values.stride(0);
    int64_t r_i = 0, t_i = 0, s_i = 0;
    auto t_indices = t._indices();
    auto src_indices = src._indices();

    // NB: relies on nnz tests above
    auto t_indices_accessor = t_indices.accessor<int64_t, 2>();
    auto r_indices_accessor = r_indices.accessor<int64_t, 2>();
    auto src_indices_accessor = src_indices.accessor<int64_t, 2>();

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
        commonDtype, "cadd_sparse", [&] {
          scalar_t* t_values_ptr = t_values.data_ptr<scalar_t>();
          scalar_t* s_values_ptr = s_values.data_ptr<scalar_t>();
          scalar_t* r_values_ptr = r_values.data_ptr<scalar_t>();
          scalar_t cast_value = value.to<scalar_t>();
          while (t_i < t_nnz || s_i < s_nnz) {
            int64_t cmp;
            if (t_i >= t_nnz) {
              cmp = -1;
            } else if (s_i >= s_nnz) {
              cmp = 1;
            } else {
              cmp = 0;
              for (auto d: c10::irange(sparse_dim)) {
                if (t_indices_accessor[d][t_i] < src_indices_accessor[d][s_i]) {
                  cmp = 1;
                  break;
                }
                if (t_indices_accessor[d][t_i] > src_indices_accessor[d][s_i]) {
                  cmp = -1;
                  break;
                }
              }
            }
            if (cmp >= 0) {
              for (auto d: c10::irange(sparse_dim)) {
                r_indices_accessor[d][r_i] = t_indices_accessor[d][t_i];
              }
              if (t_values.numel() > 0) {  // We add all elements from t_values to r_values only if t_values is not an empty tensor
                at::native::cpublas::axpy<scalar_t>(blockSize, 1,
                  t_values_ptr + t_i * blockSize, 1,
                  r_values_ptr + r_i * blockSize, 1);
              }
              t_i++;
            }
            if (cmp <= 0) {
              for (auto d: c10::irange(sparse_dim)) {
                r_indices_accessor[d][r_i] = src_indices_accessor[d][s_i];
              }
              if (s_values.numel() > 0) {  // We add all elements from s_values to r_values only if s_values is not an empty tensor
                at::native::cpublas::axpy<scalar_t>(blockSize, cast_value,
                  s_values_ptr + s_i * blockSize, 1,
                  r_values_ptr + r_i * blockSize, 1);
              }
              s_i++;
            }
            r_i++;
          }
        }
    );

    if (r.scalar_type() != commonDtype) {
      r_values = r_values.to(r.scalar_type());
    }
    get_sparse_impl(r)->set_indices_and_values_unsafe(r_indices, r_values);
    get_sparse_impl(r)->set_nnz_and_narrow(r_i);

    // TODO: I think it may be possible to track inside the loop and
    // detect when we are uncoalesced (e.g., by observing that an
    // index goes backwards) which may be more precise than using the
    // coalesced flag here.  But this is easy.
    return r._coalesced_(coalesced);
}

SparseTensor& add_out_sparse_non_contiguous(SparseTensor& r, const SparseTensor& t, const SparseTensor& src, const Scalar& value, ScalarType commonDtype) {
    Tensor t_values = t._values().to(commonDtype);
    Tensor s_values = src._values().to(commonDtype);

    // If `t` or `src` contains non-contiguous `values`, `at::native::cpublas::axpy` doesn't work
    // and we concat the indices and values tensors instead.
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
      commonDtype, "add_out_sparse_cpu", [&] {
          if (value.to<scalar_t>() != static_cast<scalar_t>(1)) {
            s_values = s_values.mul(value);
          }
        });

    Tensor r_indices = at::cat({t._indices(), src._indices()}, 1);
    Tensor r_values = at::cat({t_values, s_values}, 0).to(r.scalar_type());
    alias_into_sparse(r, r_indices, r_values);

    // Prevent unbounded growth of nnz
    // TODO: Improved heuristic on when to coalesce or remove need to coalesce
    if (r._nnz() > r.numel()) {
      auto c = r.coalesce();
      alias_into_sparse(r, c._indices(), c._values());
    }

    return r;
}

Tensor& add_out_dense_sparse_cpu(Tensor& r, const Tensor& dense, const SparseTensor& sparse_, const Scalar& value);

SparseTensor& add_out_sparse_cpu(const SparseTensor& t, const SparseTensor& src, const Scalar& value, SparseTensor& r) {
  if (!t.is_sparse()) {
    return add_out_dense_sparse_cpu(r, t, src, value);
  }
  // TODO: This test seems a bit goofy
  TORCH_CHECK(src.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  AT_ASSERT(!t.is_cuda());  // the dispatch argument
  TORCH_CHECK(!r.is_cuda(), "add: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!src.is_cuda(), "add: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(t.sizes().equals(src.sizes()), "add: expected sizes of 'self' and 'other' to match, but ", t.sizes(), " != ", src.sizes());

  auto commonDtype = promoteTypes(t.scalar_type(), src.scalar_type());

  TORCH_CHECK(canCast(commonDtype, r.scalar_type()), "Can't convert result type ", commonDtype, " to output ", r.scalar_type(), " in add operation");

  if (src._nnz() == 0) {
    return copy_sparse_to_sparse_(r, t);
  }
  if (t._nnz() == 0) {
    return mul_out_sparse_scalar(r, src, value);
  }

  TORCH_CHECK(is_same_density(t, src), "add: expected 'self' and 'other' to have same density, but 'self' has ", t.sparse_dim(), " sparse dimensions while 'other' has ", src.sparse_dim(), " sparse dimensions");

  r.resize_as_(src);

  if (src._values().is_contiguous() && t._values().is_contiguous()) {
    return add_out_sparse_contiguous(r, t, src, value, commonDtype);
  } else {
    return add_out_sparse_non_contiguous(r, t, src, value, commonDtype);
  }
}

// --------------------------------------------------------------------
// add(Tensor, SparseTensor, Scalar)
//    formerly known as spcadd
// --------------------------------------------------------------------

template <typename scalar_t>
void add_dense_sparse_worker_cpu(Tensor& r, const Scalar& value, const SparseTensor& sparse, const Tensor& indices, const Tensor& values) {
  auto indices_accessor = indices.accessor<int64_t, 2>();
  auto values_accessor = values.accessor<scalar_t, 1>();

  scalar_t* r_ptr = r.data_ptr<scalar_t>();
  auto r_strides = r.strides();
  scalar_t cast_value = value.to<scalar_t>();
  const auto sparse_dim = sparse.sparse_dim();

  at::parallel_for(0, sparse._nnz(), 0, [&](int64_t start, int64_t end) {
    for (auto k: c10::irange(start, end)) {
      int64_t index = r.storage_offset();
      for (auto d: c10::irange(sparse_dim)) {
        index += r_strides[d] * indices_accessor[d][k];
      }
      r_ptr[index] += cast_value * values_accessor[k];
    }
  });
}

Tensor& add_out_dense_sparse_cpu(Tensor& r, const Tensor& dense, const SparseTensor& sparse_, const Scalar& value) {
  AT_ASSERT(!r.is_sparse());
  AT_ASSERT(!dense.is_sparse());
  AT_ASSERT(sparse_.is_sparse());

  AT_ASSERT(!dense.is_cuda()); // dispatch argument
  TORCH_CHECK(!r.is_cuda(), "add: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!sparse_.is_cuda(), "add: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(dense.sizes().equals(sparse_.sizes()), "add: expected 'self' and 'other' to have same size, but self has size ",
    dense.sizes(), " while other has size ", sparse_.sizes(), " (FYI: dense-sparse addition does not currently support broadcasting)");

  auto commonDtype = promoteTypes(dense.scalar_type(), sparse_.scalar_type());
  TORCH_CHECK(canCast(commonDtype, r.scalar_type()), "Can't convert result type ", commonDtype, " to output ", r.scalar_type(), " in add operation");

  r.resize_as_(dense);
  SparseTensor sparse = sparse_.coalesce();

  Tensor indices = sparse._indices();
  Tensor values = sparse._values();
  int64_t nDim = dense.dim();
  int64_t nDimI = sparse.sparse_dim();

  if (sparse._nnz() == 0) {
    if (!is_same_tensor(r, dense)) r.copy_(dense);
    return r;
  }

  Tensor valuesBuffer = values.to(commonDtype);
  Tensor resultBuffer = r;
  if (r.scalar_type() != commonDtype) {
    resultBuffer = dense.to(commonDtype);
  } else if (!is_same_tensor(r, dense)) {
    resultBuffer.copy_(dense);
  }

  // accessors rely on nnz test
  if (nDim > nDimI) {
    auto indices_accessor = indices.accessor<int64_t, 2>();
    for (const auto k : c10::irange(sparse._nnz())) {
      Tensor dstBuffer = resultBuffer;
      for (const auto d : c10::irange(sparse.sparse_dim())) {
        dstBuffer = dstBuffer.select(0, indices_accessor[d][k]);
      }
      Tensor srcBuffer = valuesBuffer.select(0, k);
      dstBuffer.add_(srcBuffer, value);
    }
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::Bool,
        commonDtype, "add_dense_sparse", [&] {
          add_dense_sparse_worker_cpu<scalar_t>(resultBuffer, value, sparse, indices, valuesBuffer);
        });
  }
  if (r.scalar_type() != commonDtype) {
    r.copy_(resultBuffer);
  }
  return r;
}

// --------------------------------------------------------------------
// mul(SparseTensor, SparseTensor)  [broadcasts]
// --------------------------------------------------------------------

Tensor mul_sparse(const Tensor& self, const Tensor& other) {
  auto commonDtype = at::result_type(self, other);
  // Arbitrary (dense, sparse) and (sparse, dense) multiplication is not
  // currently supported, but (0dim-dense, sparse) and (sparse, 0dim-dense) is.
  // Make sure we use the sparse exemplar for result.
  auto result_options = self.is_sparse() ?
    self.options().dtype(commonDtype) : other.options().dtype(commonDtype);
  Tensor result = at::empty({0}, result_options);
  return at::mul_out(result, self, other);  // redispatch!
}

Tensor& mul_sparse_(Tensor& self, const Tensor& other) {
  return at::mul_out(self, self, other);  // redispatch!
}

SparseTensor& mul_out_sparse_cpu(const Tensor& t_, const Tensor& src_, SparseTensor& r) {
  if (src_.dim() == 0) {
    return mul_out_sparse_zerodim(r, t_, src_);
  } else if (t_.dim() == 0) {
    return mul_out_sparse_zerodim(r, src_, t_);
  }

  TORCH_CHECK(t_.sizes().equals(src_.sizes()), "mul operands have incompatible sizes");
  AT_ASSERT(!t_.is_cuda()); // dispatch argument
  TORCH_CHECK(!r.is_cuda(), "mul: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!src_.is_cuda(), "mul: expected 'other' to be a CPU tensor, but got a CUDA tensor");
  TORCH_CHECK(src_.is_sparse(), "mul(sparse, dense) is not supported");
  TORCH_CHECK(t_.is_sparse(), "mul(dense, sparse) is not supported");
  TORCH_CHECK(t_.sizes().equals(src_.sizes()), "mul: expected 'self' and 'other' to have same sizes, but ", t_.sizes(), " != ", src_.sizes());

  if (src_._nnz() == 0 || t_._nnz() == 0) {
    r.resize_as_(src_);
    return r.zero_();
  }

  SparseTensor t = t_.coalesce();
  SparseTensor src = src_.coalesce();

  // saving those because they can be overwritten when doing in-place operations
  int64_t t_nnz = t._nnz(), s_nnz = src._nnz();
  int64_t max_nnz = std::min(t_nnz, s_nnz);  // multiply by zero is zero, and can be dropped
  int64_t sparse_dim = src.sparse_dim();
  Tensor t_indices = t._indices();
  Tensor src_indices = src._indices();
  Tensor r_indices = at::empty({sparse_dim, max_nnz}, t_indices.options());

  int64_t r_i = 0, t_i = 0, s_i = 0;

  auto commonDtype = promoteTypes(t_.scalar_type(), src_.scalar_type());
  TORCH_CHECK(canCast(commonDtype, r.scalar_type()), "Can't convert result type ", commonDtype, " to output ", r.scalar_type(), " in mul operation");

  Tensor t_values = t._values().to(commonDtype);
  Tensor s_values = src._values().to(commonDtype);

  Tensor r_buffer = new_values_with_size_of(t_values, max_nnz).zero_();

  // NB: relies on nnz test above
  auto t_indices_accessor = t_indices.accessor<int64_t, 2>();
  auto r_indices_accessor = r_indices.accessor<int64_t, 2>();
  auto src_indices_accessor = src_indices.accessor<int64_t, 2>();

  // Check if we can find matching indices, and if so, write an
  // entry to the result indices vector.  Returns true if matching
  // indices were found.
  auto index_preamble = [&]() {
    for (auto d: c10::irange(sparse_dim)) {
      if (t_indices_accessor[d][t_i] < src_indices_accessor[d][s_i]) {
        t_i++;
        return false;
      }
      if (t_indices_accessor[d][t_i] > src_indices_accessor[d][s_i]) {
        s_i++;
        return false;
      }
    }
    for (auto d: c10::irange(sparse_dim)) {
      r_indices_accessor[d][r_i] = t_indices_accessor[d][t_i];
    }
    return true;
  };

  if (t_values.dim() > 1) {
    while (t_i < t_nnz && s_i < s_nnz) {
      if (!index_preamble()) continue;
      r_buffer.select(0, r_i).addcmul_(t_values.select(0, t_i), s_values.select(0, s_i));
      r_i++;
      t_i++;
      s_i++;
    }
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
        commonDtype, "mul_out_sparse", [&] {
          auto r_accessor = r_buffer.accessor<scalar_t, 1>();
          auto t_accessor = t_values.accessor<scalar_t, 1>();
          auto s_accessor = s_values.accessor<scalar_t, 1>();

          while (t_i < t_nnz && s_i < s_nnz) {
            if (!index_preamble()) continue;
            r_accessor[r_i] = t_accessor[t_i] * s_accessor[s_i];
            r_i++;
            t_i++;
            s_i++;
          }
        }
    );
  }

  r.resize_as_(src);
  Tensor r_values = r_buffer.to(r.scalar_type());
  get_sparse_impl(r)->set_indices_and_values_unsafe(r_indices, r_values);
  get_sparse_impl(r)->set_nnz_and_narrow(r_i);
  return r._coalesced_(true);
}

// --------------------------------------------------------------------
// addmm(D1, S, D2, beta, alpha) -> D  [broadcasts]
//
// D = beta * D1 + alpha * mm(S, D2)
// --------------------------------------------------------------------

template <typename scalar_t>
void s_addmm_out_sparse_dense_worker(int64_t nnz, int64_t dim_i, int64_t dim_j, int64_t dim_k, Tensor& r, const Scalar& beta, const Tensor& t, const Scalar& alpha, const Tensor& indices, const Tensor& values, const Tensor& dense) {

  // r_ = alpha * sparse * dense
  scalar_t cast_alpha = alpha.to<scalar_t>();
  scalar_t cast_beta = beta.to<scalar_t>();

  if (cast_beta == static_cast<scalar_t>(0)) {
    r.zero_();
  } else if (cast_beta == static_cast<scalar_t>(1)) {
    if (!is_same_tensor(r, t)) {
      r.copy_(t);
    }
  } else {
    at::mul_out(r, t, scalar_to_tensor(beta));
  }

  auto indices_accessor = indices.accessor<int64_t, 2>();

  auto values_accessor = values.accessor<scalar_t, 1>();
  scalar_t* dense_ptr = dense.data_ptr<scalar_t>();
  scalar_t* r_ptr = r.data_ptr<scalar_t>();

  int64_t dense_stride0 = dense.stride(0);
  int64_t dense_stride1 = dense.stride(1);
  int64_t r_stride0 = r.stride(0);
  int64_t r_stride1 = r.stride(1);
  for (auto i: c10::irange(nnz)) {
    scalar_t val = values_accessor[i];
    int64_t row = indices_accessor[0][i];
    int64_t col = indices_accessor[1][i];
    if (col >= 0 && col < dim_j && row >= 0 && row < dim_i) {
      at::native::cpublas::axpy<scalar_t>(dim_k,
            cast_alpha * val,
            dense_ptr + col * dense_stride0, dense_stride1,
            r_ptr + row * r_stride0, r_stride1);
    } else {
      if (col < 0 || col >= dim_j) {
        AT_ERROR("addmm: index out of column bound: ", col, " not between 1 and ", dim_j);
      } else {
        AT_ERROR("addmm: index out of row bound: ", row, " not between 1 and ", dim_i);
      }
    }
  }
};

Tensor& s_addmm_out_sparse_dense_cpu(
    Tensor& r,
    const Tensor& t,
    const SparseTensor& sparse_,
    const Tensor& dense,
    const Scalar& beta,
    const Scalar& alpha
) {
  // TODO: This error message seems awfully opaque
  TORCH_CHECK(!t.is_cuda(),  "Expected all tensors to be on the same device. addmm expected 't' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!r.is_cuda(), "Expected all tensors to be on the same device. addmm: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!sparse_.is_cuda(), "Expected all tensors to be on the same device. addmm: expected 'mat1' to be a CPU tensor, but got a CUDA tensor");
  TORCH_CHECK(!dense.is_cuda(), "Expected all tensors to be on the same device. addmm: expected 'mat2' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(sparse_.sparse_dim() == 2, "addmm: matrices expected, got ", sparse_.sparse_dim(), "D tensor");
  TORCH_CHECK(sparse_.dense_dim() == 0, "addmm: scalar values expected, got ", sparse_.dense_dim(), "D values");
  TORCH_CHECK(dense.dim() == 2, "addmm: matrices expected, got ", dense.dim(), "D tensor");

  // ixj * jxk = ixk
  int64_t dim_i = sparse_.size(0);
  int64_t dim_j = sparse_.size(1);
  int64_t dim_k = dense.size(1);

  TORCH_CHECK(dense.size(0) == dim_j,
      "addmm: Argument #3 (dense): Expected dim 0 size ", dim_j, ", got ", dense.size(0));
  TORCH_CHECK(t.size(0) == dim_i,
      "addmm: Argument #1 (t): Expected dim 0 size ", dim_i, ", got ", t.size(0));
  TORCH_CHECK(t.size(1) == dim_k,
      "addmm: Argument #1 (t): Expected dim 1 size ", dim_k, ", got ", t.size(1));

  r.resize_({dim_i, dim_k});

  int64_t nnz        = sparse_._nnz();

  if (nnz == 0) {
    at::mul_out(r, t, at::scalar_tensor(beta, r.options()));
    return r;
  }

  Tensor indices = sparse_._indices();
  Tensor values      = sparse_._values();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
      values.scalar_type(), "addmm_sparse_dense", [&] {
        s_addmm_out_sparse_dense_worker<scalar_t>(nnz, dim_i, dim_j, dim_k, r, beta, t, alpha, indices, values, dense);
      }
  );

  return r;

}

Tensor& addmm_out_sparse_dense_cpu(
    const Tensor& self,
    const SparseTensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  c10::MaybeOwned<Tensor> b_self = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  return s_addmm_out_sparse_dense_cpu(result, *b_self, mat1, mat2, beta, alpha);
}

Tensor s_addmm_sparse_dense_cpu(
    const Tensor& t,
    const SparseTensor& sparse,
    const Tensor& dense,
    const Scalar& beta,
    const Scalar& alpha
) {
  Tensor r = at::empty({0}, t.options());
  s_addmm_out_sparse_dense_cpu(r, t, sparse, dense, beta, alpha);
  return r;
}

Tensor addmm_sparse_dense_cpu(
    const Tensor& self,
    const SparseTensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha
) {
  c10::MaybeOwned<Tensor> b_self = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  return s_addmm_sparse_dense_cpu(*b_self, mat1, mat2, beta, alpha);
}

Tensor& s_addmm_sparse_dense_cpu_(
    Tensor& t,
    const SparseTensor& sparse,
    const Tensor& dense,
    const Scalar& beta,
    const Scalar& alpha
) {
  return s_addmm_out_sparse_dense_cpu(t, t, sparse, dense, beta, alpha);
}

// NB: Purposely no broadcasting version of addmm inplace

Tensor _sparse_addmm(
  const Tensor& t,
  const SparseTensor& sparse,
  const Tensor& dense,
  const Scalar& beta,
  const Scalar& alpha
) {
  // _sparse_addmm forward is functionally equivalent to addmm; it's
  // just the backward that is different.  This technically does an
  // unnecessary redispatch, I was too lazy to make it not do that
  return at::addmm(t, sparse, dense, beta, alpha);
}

Tensor _sparse_mm(
  const SparseTensor& sparse,
  const Tensor& dense
) {
  Tensor t = at::zeros({}, dense.options());
  return at::_sparse_addmm(t, sparse, dense, 0, 1);  // redispatch!
}

// NB: Despite its suggestive name, this actually only exists so that
// we can redispatch to addmm_out; this is NOT an implementation of
// the sparse masking version of mm
SparseTensor& _sparse_mm_out(const SparseTensor& sparse,
  const Tensor& dense,
  SparseTensor& result) {
  Tensor t = at::zeros({}, dense.options());
  return at::addmm_out(result, t, sparse, dense, 0, 1);  // redispatch!
}

// --------------------------------------------------------------------
// hspmm(SparseTensor mat1, Tensor mat2)
// --------------------------------------------------------------------

SparseTensor& hspmm_out_sparse_cpu(const SparseTensor& sparse_, const Tensor& dense, SparseTensor& r) {
  // TODO: Make this a real argument
  Scalar alpha = 1;

  AT_ASSERT(!sparse_.is_cuda()); // dispatch argument
  TORCH_CHECK(!r.is_cuda(), "hspmm: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!dense.is_cuda(), "hspmm: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(sparse_.sparse_dim() == 2,
      "hspmm: Argument #2: matrices expected, got ", sparse_.sparse_dim(), "D tensor");
  TORCH_CHECK(sparse_.dense_dim() == 0,
      "hspmm: Argument #2: scalar values expected, got ", sparse_.dense_dim(), "D values");
  TORCH_CHECK(dense.dim() == 2,
      "hspmm: Argument #3: matrices expected, got ", dense.dim(), "D tensor");

  int64_t m = sparse_.size(0);
  int64_t k = sparse_.size(1);
  int64_t n = dense.size(1);

  TORCH_CHECK(dense.size(0) == k,
      "hspmm: Argument #3: Expected dim 0 size ", k, ", got ", dense.size(0));

  get_sparse_impl(r)->raw_resize_(1, 1, {m, n});

  SparseTensor sparse = sparse_.coalesce();

  int64_t nnz = sparse._nnz();

  if (nnz == 0) {
    r.zero_();
    return r;
  }

  Tensor indices = at::empty({1, nnz}, at::initialTensorOptions().dtype(kLong));

  // Initialize the sparse matrix that will be used with spaddmm to send rows
  // from the dense matrix to rows of the output's value tensor
  SparseTensor newSparse = sparse.clone();
  Tensor spIndices = newSparse._indices();
  Tensor valueIndices = spIndices.select(0, 0);

  // Compute output indices
  auto valueIndices_accessor = valueIndices.accessor<int64_t, 1>();
  auto indices_accessor = indices.accessor<int64_t, 2>();

  int64_t i = -1, prevIdx = -1;
  for (const auto j : c10::irange(nnz)) {
    int64_t currIdx = valueIndices_accessor[j];
    if (currIdx != prevIdx) {
      indices_accessor[0][++i] = currIdx;
      prevIdx = currIdx;
    }
    valueIndices_accessor[j] = i;
  }
  int64_t outNnz = i + 1;
  indices.resize_({1, outNnz});
  Tensor values = at::empty({outNnz, n}, dense.options());

  std::vector<int64_t> new_size = get_sparse_impl(newSparse)->sizes().vec();
  new_size[0] = outNnz;
  get_sparse_impl(newSparse)->raw_resize_(get_sparse_impl(newSparse)->sparse_dim(), get_sparse_impl(newSparse)->dense_dim(), new_size);

  // Compute output values tensor with sparse * dense multiplication
  s_addmm_out_sparse_dense_cpu(values, values, newSparse, dense, 0, alpha);
  get_sparse_impl(r)->set_indices_and_values_unsafe(indices, values);

  return r;
}

SparseTensor hspmm_sparse_cpu(const SparseTensor& sparse, const Tensor& dense) {
  SparseTensor r = at::empty({0}, sparse.options());
  hspmm_out_sparse_cpu(sparse, dense, r);
  return r;
}

// --------------------------------------------------------------------
// sspaddmm(S1, S2, D, beta, alpha) -> S
//
// S = beta * S1 + alpha * mm(S2, D)
// --------------------------------------------------------------------

SparseTensor& _sspaddmm_out_cpu(
    const SparseTensor& t,
    const SparseTensor& sparse_,
    const Tensor& dense,
    const Scalar& beta,
    const Scalar& alpha,
    SparseTensor& r) {
  AT_ASSERT(!t.is_cuda()); // dispatch argument
  TORCH_CHECK(!r.is_cuda(), "sspaddmm: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!sparse_.is_cuda(), "sspaddmm: expected 'mat1' to be a CPU tensor, but got a CUDA tensor");
  TORCH_CHECK(!dense.is_cuda(), "sspaddmm: expected 'mat2' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(sparse_.sparse_dim() == 2,
      "sspaddmm: Argument #2: matrices expected, got ", sparse_.sparse_dim(), "D tensor");
  TORCH_CHECK(sparse_.dense_dim() == 0,
      "sspaddmm: Argument #2: scalar values expected, got ", sparse_.dense_dim(), "D values");
  TORCH_CHECK(dense.dim() == 2,
      "sspaddmm: Argument #2: matrices expected, got ", dense.dim(), "D tensor");

  SparseTensor sparse = sparse_.coalesce();

  // ixj * jxk = ixk
  int64_t dim_i = sparse.size(0);
  int64_t dim_j = sparse.size(1);
  int64_t dim_k = dense.size(1);

  // NB: This has to occur before the checks, because r may alias t.
  // See test_saddmm
  get_sparse_impl(r)->raw_resize_(2, 0, {dim_i, dim_k});

  TORCH_CHECK(dense.size(0) == dim_j,
      "sspaddmm: Argument #3: Expected dim 0 size ", dim_j, ", got ", dense.size(0));
  TORCH_CHECK(t.size(0) == dim_i,
      "sspaddmm: Argument #1: Expected dim 0 size ", dim_i, ", got ", t.size(0));
  TORCH_CHECK(t.size(1) == dim_k,
      "sspaddmm: Argument #1: Expected dim 1 size ", dim_k, ", got ", t.size(1));

  int64_t nnz        = sparse._nnz();
  // We have to make indices contiguous as we use indices.data_ptr in _to_csr which assumes row-contiguous storage
  Tensor indices = sparse._indices().contiguous();
  Tensor values      = sparse._values();

  Tensor csr = coo_to_csr(indices.data_ptr<int64_t>(), dim_i, nnz);

  int64_t t_nnz = t._nnz();
  int64_t r_nnz = nnz * dim_k + t_nnz;
  Tensor newi = at::empty({2, r_nnz}, kLong);
  Tensor newv = native::zeros(
      {r_nnz},
      optTypeMetaToScalarType(values.options().dtype_opt()),
      values.options().layout_opt(),
      values.options().device_opt(),
      values.options().pinned_memory_opt());

  if (t_nnz != 0) {
    Tensor narrowi = newi.narrow(1, 0, t_nnz);
    Tensor narrowv = newv.narrow(0, 0, t_nnz);

    narrowi.copy_(t._indices());
    narrowv.copy_(t._values());
    newv.mul_(beta);
  }

  // sparse = sparse * dense
  int64_t p = t_nnz;

  auto csr_accessor = csr.accessor<int64_t, 1>();
  auto indices_accessor = indices.accessor<int64_t, 2>();
  auto newi_accessor = newi.accessor<int64_t, 2>();

  int64_t dense_stride0 = dense.stride(0);
  int64_t dense_stride1 = dense.stride(1);
  int64_t newv_stride0 = newv.stride(0);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
      values.scalar_type(), "sspmm", [&] {
        auto values_accessor = values.accessor<scalar_t, 1>();
        scalar_t* dense_ptr = dense.data_ptr<scalar_t>();
        scalar_t* newv_ptr = newv.data_ptr<scalar_t>();
        scalar_t cast_alpha = alpha.to<scalar_t>();

        for (const auto h : c10::irange(dim_i)) {
          int64_t i_start = csr_accessor[h];
          int64_t i_end = csr_accessor[h+1];
          for (const auto i : c10::irange(i_start, i_end)) {
            scalar_t val = values_accessor[i];
            int64_t col = indices_accessor[1][i];
            if (col >= 0 && col < dim_j) {
              at::native::cpublas::axpy<scalar_t>(dim_k,
                  cast_alpha * val,
                  dense_ptr + col * dense_stride0, dense_stride1,
                  newv_ptr + p * newv_stride0, 1);
            } else {
              AT_ERROR("index out of bound. sspmm: ", col, " not between 1 and ", dim_j);
            }
          }
          // Fill up the indices with the right values
          if (i_start != i_end) {
            for (const auto i : c10::irange(dim_k)) {
              newi_accessor[0][p+i] = h;
              newi_accessor[1][p+i] = i;
            }
            p += dim_k;
          }
        }
      }
  );

  // to avoid a clone
  get_sparse_impl(r)->set_indices_and_values_unsafe(newi, newv);
  get_sparse_impl(r)->set_nnz_and_narrow(p);

  return r;
}

// sparse, sparse, sparse, dense, real, real -> sparse
Tensor& _sspaddmm_out_only_sparse(const Tensor& self,
    const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, Tensor& result) {
  AT_ERROR("tensor.sspaddmm(...) can only be called on sparse tensors");
}

// sparse, dense -> sparse
Tensor smm(const Tensor& self, const Tensor& mat2) {
  auto result = at::empty({0}, self.options());
  at::sspaddmm_out(result, result, self, mat2, 0.0, 1.0);
  return result;
}

// sparse, sparse, dense, real, real -> sparse
Tensor sspaddmm(const Tensor& self, const Tensor& mat1, const Tensor& mat2,
    const Scalar& beta, const Scalar& alpha) {
  auto result = at::empty({0}, self.options());
  at::sspaddmm_out(result, self, mat1, mat2, beta, alpha);
  return result;
}

// --------------------------------------------------------------------
// sparse.sum()
//
// This implementation calls coalesce() to do the sum reduction on
// sparse dims. Ideally in the future there should be unified reduction function
// for ops like sum, max, and min.
// --------------------------------------------------------------------
Tensor _sparse_sum(const SparseTensor& input) {
  return input.coalesce().values().sum();
}

Tensor _sparse_sum(const SparseTensor& input, ScalarType dtype) {
  // don't have to do a conversion to the correct dtype first
  // just need to setup the accumulator correctly
  return input.coalesce().values().sum(dtype);
}

Tensor _sparse_sum(const SparseTensor& input, IntArrayRef dims_to_sum, ScalarType dtype) {
  return at::_sparse_sum(input.to(dtype), dims_to_sum);
}

Tensor _sparse_sum(const SparseTensor& input, IntArrayRef dims_to_sum) {
  const int64_t input_dim = input.dim();
  auto dims_to_sum_b = dim_list_to_bitset(dims_to_sum, input_dim);
  auto dims_to_sum_v = dims_to_sum.vec();
  maybe_wrap_dims(dims_to_sum_v, input_dim);

  Tensor indices = input._indices();
  Tensor values = input._values();
  IntArrayRef sizes = input.sizes();
  const int64_t sparse_dim = input.sparse_dim();

  auto dims_to_keep_v = std::vector<int64_t>();
  auto dense_dims_to_sum_v = std::vector<int64_t>();
  for (const auto d : c10::irange(input_dim)) {
    if (dims_to_sum_b[d]) {
      if (d >= sparse_dim) dense_dims_to_sum_v.emplace_back(d + 1 - sparse_dim);
    }
    else {
      dims_to_keep_v.emplace_back(d);
    }
  }
  const int64_t sparse_dims_to_sum_size = dims_to_sum_v.size() - dense_dims_to_sum_v.size();
  const bool sum_all_sparse_dim = (sparse_dim == sparse_dims_to_sum_size);
  const bool sum_dense_dim = (dense_dims_to_sum_v.size() > 0);

  // new values
  Tensor new_values;
  if (sum_dense_dim) {
    new_values = values.sum(dense_dims_to_sum_v);
  }
  else {
    new_values = values.clone(at::MemoryFormat::Contiguous);
  }

  if (sum_all_sparse_dim) {
    // return a dense tensor if sum over all sparse dims
    new_values = new_values.sum(0);
    return new_values;
  }
  else { // !sum_all_sparse_dim
    // new indices
    Tensor new_indices;
    if (sparse_dims_to_sum_size == 0) {
      new_indices = indices.clone(at::MemoryFormat::Contiguous);
    }
    else {
      new_indices = at::empty({sparse_dim - sparse_dims_to_sum_size, input._nnz()}, indices.options());
      for (auto i: c10::irange(dims_to_keep_v.size())) {
        int64_t d = dims_to_keep_v[i];
        if (d < sparse_dim) new_indices[i].copy_(indices[d]);
        else break;
      }
    }

    // new size
    int64_t new_sparse_dim = new_indices.size(0);
    int64_t new_dense_dim = new_values.dim() - 1; // exclude nnz dim
    std::vector<int64_t> new_sizes;
    new_sizes.reserve(dims_to_keep_v.size());
    for (auto d : dims_to_keep_v) new_sizes.emplace_back(sizes[d]);
    if (sum_all_sparse_dim) new_sizes.emplace(new_sizes.begin(), 1);

    // use coalesce() to do sum reduction
    SparseTensor new_sparse = at::_sparse_coo_tensor_with_dims_and_tensors(new_sparse_dim, new_dense_dim, new_sizes, new_indices, new_values, input.options());
    new_sparse = new_sparse.coalesce();
    return new_sparse;
  }

}
// --------------------------------------------------------------------
// NOTE [ sparse.sum() backward ]
//
// When sum over sparse_dim, backward scatters gradients from grad tensor to input tensor.
// Grad and input need to align indices over sparse_dim that are not summed (given
// input.spares_dim >= grad.sparse_dim). Implementation here compares each pair of
// indices between grad and input. When a matching indices pair (input_i, grad_i) is found,
// copy grad.values[grad_i] -> input_grad.values[input_i]. E.g.,
//
//  input.sparse_dim = [5, 5]
//  input.indices = [[0, 0, 1, 2, 2, 3, 4, 4],
//                   [1, 4, 4, 0, 1, 3, 2, 4]]
//  input.values =   [0, 1, 2, 3, 4, 5, 6, 7]
//  ...
//  sparse.sum(input, [0])
//  backward(...)
//  ...
//  grad.indices = [[0, 1, 2, 3]]
//  grad.values =   [1, 2, 0, 4]
//
// # after indices matching
//         input         grad
//        [[0, 1],   ->  [1]
//         [0, 4],   ->  [ ]
//         [1, 4],   ->  [ ]
//         [2, 0],   ->  [0]
//         [2, 1],   ->  [1]
//         [3, 3],   ->  [3]
//         [4, 2],   ->  [2]
//         [4, 4]])  ->  [ ]
//
// input_grad.indices = [[0, 0, 1, 2, 2, 3, 4, 4],
//                       [1, 4, 4, 0, 1, 3, 2, 4]]
// input_grad.values =   [2, 0, 0, 1, 2, 4, 0, 0]
//
// Note that we allow input to be uncoalesced in the forward,
// we have to coalesce input at the backward, because grad-of-input
// take the same indices as input, if input is not coalesced, then
// coalescing grad-of-input may add up grad values for a duplicate indices,
// and hence generates a wrong grad-of-input.
//
// Other edge cases:
// - assign zero values to input gradients if cannot find matched indices at grad
// - grad.values might have zeros
// --------------------------------------------------------------------
Tensor _sparse_sum_backward_cpu(const Tensor& grad_, const SparseTensor& input_, IntArrayRef dims_to_sum) {
  TORCH_CHECK(!grad_.is_cuda(), "_sparse_sum_backward_cpu: expected 'grad_' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!input_.is_cuda(), "_sparse_sum_backward_cpu: expected 'input_' to be CPU tensor, but got CUDA tensor");

  auto input = input_.coalesce();
  const int64_t input_dim = input.dim();
  auto dims_to_sum_b = dim_list_to_bitset(dims_to_sum, input_dim);
  auto dims_to_sum_v = dims_to_sum.vec();
  maybe_wrap_dims(dims_to_sum_v, input_dim);

  Tensor input_indices = input._indices();
  Tensor input_values = input._values();
  IntArrayRef input_sizes = input.sizes();
  const int64_t input_sparse_dim = input.sparse_dim();
  const int64_t input_dense_dim = input.dense_dim();
  const int64_t input_nnz = input._nnz();

  int64_t sparse_dims_to_sum_size = 0;
  auto sparse_dims_to_keep_v = std::vector<int64_t>();
  auto dense_dims_to_sum_v = std::vector<int64_t>();
  for (auto d: c10::irange(input_dim)) {
    if (dims_to_sum_b[d]) {
      if (d < input_sparse_dim) sparse_dims_to_sum_size ++;
      else dense_dims_to_sum_v.emplace_back(d + 1 - input_sparse_dim);
    }
    else {
      if (d < input_sparse_dim) sparse_dims_to_keep_v.emplace_back(d);
    }
  }

  const bool sum_all_sparse_dim = (input_sparse_dim == sparse_dims_to_sum_size);
  const bool sum_dense_dim = (dense_dims_to_sum_v.size() > 0);
  const bool sum_sparse_dim = (sparse_dims_to_sum_size > 0);

  if (sum_all_sparse_dim) {
    TORCH_CHECK(!grad_.is_sparse(), "_sparse_sum_backward_cpu: expected grad_ Tensor to be dense since all sparse dims are summed");
    auto grad_input_values = grad_;
    auto expand_size = input_values.sizes().vec();
    if (sum_dense_dim) {
      auto dense_expand_size = std::vector<int64_t>(expand_size);
      dense_expand_size.erase(dense_expand_size.begin());
      AT_ASSERT(dense_expand_size.size() == static_cast<size_t>(input_values.dim() - 1));
      for (auto d : dense_dims_to_sum_v) grad_input_values = grad_input_values.unsqueeze(d - 1);  // -1 since grad has no nnz dim
      grad_input_values = grad_input_values.expand(dense_expand_size);
    }
    grad_input_values = grad_input_values.expand(expand_size).clone(at::MemoryFormat::Contiguous);
    return at::_sparse_coo_tensor_with_dims_and_tensors(input_sparse_dim, input_dense_dim, input_sizes, input_indices.clone(at::MemoryFormat::Contiguous), grad_input_values, input.options().dtype(grad_.dtype())); // convert to grad dtype
  }
  else {
    TORCH_CHECK(grad_.is_sparse(), "_sparse_sum_backward_cpu: expected grad_ Tensor to be sparse, but got dense");
    auto grad = grad_.coalesce();
    Tensor grad_indices = grad._indices();
    Tensor grad_values = grad._values();
    const int64_t grad_sparse_dim = grad.sparse_dim();
    const int64_t grad_nnz = grad._nnz();

    Tensor grad_values_expand = grad_values;
    if (sum_dense_dim) {
      auto expand_size = input_values.sizes().vec();
      if (sum_sparse_dim) expand_size[0] = grad_values.size(0);
      for (auto d : dense_dims_to_sum_v) grad_values_expand = grad_values_expand.unsqueeze(d);
      grad_values_expand = grad_values_expand.expand(expand_size).clone(at::MemoryFormat::Contiguous);
    }

    Tensor grad_input_values;
    if (sum_sparse_dim) {
      // see NOTE [ sparse.sum() backward ]
      grad_input_values = at::zeros_like(input_values, grad_values.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      // get flatten indices for grad and input
      auto grad_sparse_dim_to_keep_v = std::vector<int64_t>(grad_sparse_dim);
      std::iota(grad_sparse_dim_to_keep_v.begin(), grad_sparse_dim_to_keep_v.end(), 0);

      auto grad_indices_1D = flatten_indices_by_dims(grad_indices, grad.sizes(), grad_sparse_dim_to_keep_v); // flatten indices on all sparse_dim of grad, output indices is coalesced and sorted
      auto grad_indices_1D_accessor = grad_indices_1D.accessor<int64_t, 1>();
      auto input_indices_1D = flatten_indices_by_dims(input_indices, input_sizes, sparse_dims_to_keep_v);
      auto input_indices_1D_accessor = input_indices_1D.accessor<int64_t, 1>();

      const auto copy_iter = TensorIteratorConfig()
        .add_output(grad_input_values)
        .add_input(grad_values_expand)
        .resize_outputs(false)
        .declare_static_shape(grad_values_expand.sizes(), /*squash_dims=*/0)
        .build();
      const auto device_type = kCPU;

      const auto gIv_data = reinterpret_cast<char*>(grad_input_values.data_ptr());
      const auto gOv_data = reinterpret_cast<char*>(grad_values_expand.data_ptr());
      const auto gIv_stride = (grad_input_values.strides()[0] *
                               grad_input_values.element_size());
      const auto gOv_stride = (grad_values_expand.strides()[0] *
                               grad_values_expand.element_size());

      // binary search to find matching indices
      at::parallel_for(0, input_nnz, 0, [&](int64_t start, int64_t end) {
        TensorIterator copy_iter_local(copy_iter);

        for (auto i: c10::irange(start, end)) {
          int64_t input_idx = input_indices_1D_accessor[i];
          int64_t l = 0, r = grad_nnz - 1;
          while (l <= r) {
            int64_t m = l + (r - l) / 2;
            if (grad_indices_1D_accessor[m] == input_idx) {
              // grad_input_values[i].copy_(grad_values_expand[m])
              copy_iter_local.unsafe_replace_operand(0, gIv_data + i * gIv_stride);
              copy_iter_local.unsafe_replace_operand(1, gOv_data + m * gOv_stride);
              copy_stub(device_type, copy_iter_local, /*non_blocking=*/false);
              break;
            }
            if (grad_indices_1D_accessor[m] < input_idx) {
              l = m + 1;
            }
            else {
              r = m - 1;
            }
          }
        }
      });
    }
    else {
      grad_input_values = grad_values_expand;
    }
    return at::_sparse_coo_tensor_with_dims_and_tensors(input_sparse_dim, input_dense_dim, input_sizes, input_indices.clone(at::MemoryFormat::Contiguous), grad_input_values, grad.options());
  }
}

Tensor any_sparse(const Tensor& self) {
  TORCH_INTERNAL_ASSERT(self.is_sparse());

  return at::any(self._values());
}

Tensor bmm_sparse_cpu(const SparseTensor& self, const Tensor& mat2) {
  Tensor result = at::empty({}, mat2.options());
  return bmm_out_sparse_cpu(self, mat2, result);
}

// Search a sorted strided array for the rightmost instance of a value.
// Array must be sorted from lowest to highest.
// Returns the index of the found element.
// Returns by reference `found`, true if search value was found, false otherwise
template<typename scalar_t>
scalar_t binary_search_strided_rightmost(scalar_t search_val, TensorAccessor<scalar_t, 1>& sorted_arr_accessor, int64_t sorted_arr_begin_idx, int64_t length, bool* found) {
  if (length == 0) {
    *found = false;
    return -1;
  }

  int64_t left_ind = 0;
  int64_t right_ind = length - 1;
  int64_t mid_ind; // NOLINT(cppcoreguidelines-init-variables)
  bool done_searching = false;

  while (!done_searching) {
    mid_ind = (left_ind+right_ind) >> 1;
    scalar_t mid_val = sorted_arr_accessor[sorted_arr_begin_idx + mid_ind];

    if (mid_val > search_val) {
      right_ind = mid_ind-1;
    } else if((mid_val == search_val) && (
      (mid_ind == length - 1) || (sorted_arr_accessor[sorted_arr_begin_idx + mid_ind + 1] != search_val)
    )) {
      done_searching = true;
      *found = true;
    } else {
      left_ind = mid_ind+1;
    }

    if (left_ind > right_ind) {
      done_searching = true;
      *found = false;
      mid_ind = -1;
    }
  }

  return mid_ind;
}

Tensor& bmm_out_sparse_cpu(const SparseTensor& self, const Tensor& mat2, Tensor& result) {
  TORCH_CHECK(!mat2.is_sparse(), "bmm_sparse: Tensor 'mat2' must be dense");

  TORCH_CHECK(self.dense_dim() == 0, "bmm_sparse: Tensor 'self' must have 0 dense dims, but has ", self.dense_dim());
  TORCH_CHECK(self.sparse_dim() == 3, "bmm_sparse: Tensor 'self' must have 3 sparse dims, but has ", self.sparse_dim());
  TORCH_CHECK(mat2.dim() == 3, "bmm_sparse: Tensor 'mat2' must have 3 dims, but has ", mat2.dim());

  TORCH_CHECK(self.size(0) == mat2.size(0), "bmm_sparse: 'self.size(0)' and 'mat2.size(0)' must match");
  TORCH_CHECK(self.size(2) == mat2.size(1), "bmm_sparse: 'self.size(2)' and 'mat2.size(1)' must match");

  result.resize_({self.size(0), self.size(1), mat2.size(2)});

  if (self._nnz() == 0) {
    result.zero_();
    return result;
  }

  // First need to coalesce to get all of the first dimension indices
  // in order since we'll be sending each matrix into the MM operation
  SparseTensor self_coalesced = self.coalesce();

  int64_t nnz =        self_coalesced._nnz();
  Tensor indices = self_coalesced._indices();
  Tensor values =      self_coalesced._values();

  Tensor indices_dim0 = indices[0];
  auto indices_dim0_accessor = indices_dim0.accessor<int64_t, 1>();
  Tensor indices_dim1_dim2 = indices.slice(0, 1, 3);

  int64_t dim_i = self_coalesced.size(1);
  int64_t dim_j = self_coalesced.size(2);
  int64_t dim_k = mat2.size(2);

  Scalar beta = 0;
  Tensor t_dummy;
  Scalar alpha = 1;

  int64_t mat_el_begin_idx = 0;

  int64_t num_matrices = self_coalesced.size(0);

  // Iterate through each set of 2D matrices within the 3D
  // tensor inputs, performing a matrix multiply with each one.
  int64_t start_mat_num = indices_dim0_accessor[0];
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
    values.scalar_type(), "bmm_sparse_dense", [&] {
      for (int64_t cur_mat_num = 0;
        (cur_mat_num < num_matrices);
        cur_mat_num++
      ) {
        // If there are sparse matrices at the beginning or end that
        // have all zero elements, we need to zero out the result matrix.
        if ((cur_mat_num < start_mat_num) || (mat_el_begin_idx >= nnz)) {
          result[cur_mat_num].zero_();
          continue;
        }

        // Search for the range of sparse tensor elements that
        // correspond to the current matrix number. We already know
        // where the current matrix begins, so we just need to find
        // the end. The search excludes everything to the left of
        // the starting point, for best performance
        bool mat_end_found;
        int64_t mat_el_end_idx = binary_search_strided_rightmost(
          cur_mat_num,
          indices_dim0_accessor,
          mat_el_begin_idx,
          nnz-mat_el_begin_idx,
          &mat_end_found
        ) + mat_el_begin_idx;

        if (mat_end_found) {
          mat_el_end_idx++;

          // Create tensors to view just the current set of matrices
          const Tensor dense_matrix = mat2[cur_mat_num];
          Tensor result_matrix = result[cur_mat_num];
          Tensor sparse_indices = indices_dim1_dim2.slice(1, mat_el_begin_idx, mat_el_end_idx);
          Tensor sparse_values = values.slice(0, mat_el_begin_idx, mat_el_end_idx);
          int64_t sparse_nnz = mat_el_end_idx - mat_el_begin_idx;


          s_addmm_out_sparse_dense_worker<scalar_t>(
            sparse_nnz,
            dim_i, dim_j, dim_k,
            result_matrix,
            beta, t_dummy, alpha,
            sparse_indices, sparse_values,
            dense_matrix
          );
          mat_el_begin_idx = mat_el_end_idx;

        // If no elements for this sparse matrix are found, then
        // it's a zero matrix and we need to zero out the result
        } else {
          result[cur_mat_num].zero_();
        }
      }
    }
  );
  return result;
}

Tensor& conj_physical_out_sparse(const Tensor& input, Tensor& result) {
  TORCH_INTERNAL_ASSERT(input.is_sparse());
  if (!is_same_tensor(result, input)) {
    copy_sparse_to_sparse_(result, input);
  }
  if (!input.is_complex()) {
    return result;
  }
  Tensor result_values = result._values();
  at::conj_physical_out(result_values, input._values());
  return result;
}

}} // namespace at::native
