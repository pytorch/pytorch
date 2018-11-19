#include "ATen/native/ReduceOps.h"

#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/WrapDimUtilsMulti.h"
#include "ReduceOpsUtils.h"
#include "TensorIterator.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>
#include <map>

namespace at {
namespace native {

DEFINE_DISPATCH(sum_stub);
DEFINE_DISPATCH(prod_stub);
DEFINE_DISPATCH(norm_kernel);

static inline Tensor integer_upcast(const Tensor& self, optional<ScalarType> dtype) {
  ScalarType scalarType = self.type().scalarType();
  ScalarType upcast_scalarType = dtype.value_or(at::isIntegralType(scalarType) ? ScalarType::Long : scalarType);
  return self.toType(upcast_scalarType);
}

using DimMask = TensorIterator::DimMask;

static DimMask make_dim_mask(IntList dims, int ndim) {
  auto mask = DimMask();
  if (dims.empty()) {
    mask.flip();
  } else {
    for (int dim : dims) {
      mask.set(maybe_wrap_dim(dim, ndim));
    }
  }
  return mask;
}

static void allocate_reduction_result(
    Tensor& result, const Tensor& self, DimMask mask, bool keepdim,
    ScalarType dtype)
{
  auto shape = DimVector(self.sizes());
  for (int dim = shape.size() - 1; dim >= 0; dim--) {
    if (mask[dim]) {
      if (keepdim) {
        shape[dim] = 1;
      } else {
        shape.erase(shape.begin() + dim);
      }
    }
  }
  if (result.defined()) {
    result.resize_(shape);
  } else {
    result = at::empty(shape, self.type().toScalarType(dtype));
  }
}

static Tensor review_reduce_result(const Tensor& result, int ndim, DimMask mask, bool keepdim) {
  if (keepdim) {
    return result;
  }
  auto shape = DimVector(result.sizes());
  auto stride = DimVector(result.strides());
  for (int dim = 0; dim < ndim; dim++) {
    if (mask[dim]) {
      shape.insert(shape.begin() + dim, 1);
      stride.insert(stride.begin() + dim, 0);
    }
  }
  return result.as_strided(shape, stride);
}

static std::unique_ptr<TensorIterator> make_reduction(
    const char* name, Tensor& result, const Tensor& self, IntList dim,
    bool keepdim, ScalarType dtype)
{
  // check that result type and dtype match if provided
  AT_CHECK(
      !result.defined() || result.type().scalarType() == dtype,
      name, ": provided dtype must match dtype of result. Got ",
      at::toString(result.type().scalarType()),
      " and ",
      at::toString(dtype),
      ".");
  int ndim = self.dim();
  auto mask = make_dim_mask(dim, ndim);
  allocate_reduction_result(result, self, mask, keepdim, dtype);
  auto viewed_result = review_reduce_result(result, ndim, mask, keepdim);
  if (self.type().scalarType() != dtype) {
    return TensorIterator::reduce_op(viewed_result, self.to(dtype));
  }
  return TensorIterator::reduce_op(viewed_result, self);
}

static inline Tensor cumsum(const Tensor& self, int64_t dim, optional<ScalarType> dtype) {
  return at::_th_cumsum(integer_upcast(self, dtype), dim);
}

Tensor cumsum(const Tensor& self, int64_t dim, ScalarType dtype) {
  return at::native::cumsum(self, dim, optional<ScalarType>(dtype));
}

Tensor cumsum(const Tensor& self, int64_t dim) {
  return at::native::cumsum(self, dim, c10::nullopt);
}

static inline Tensor& cumsum_out(Tensor& result, const Tensor& self, int64_t dim, optional<ScalarType> dtype) {
  // result type is favored over dtype; check that they match if provided (NumPy doesn't check)
  AT_CHECK(
      !dtype.has_value() || (result.type().scalarType() == dtype.value()),
      "provided dtype must match dtype of result in cumsum. Got ",
      at::toString(result.type().scalarType()),
      " and ",
      at::toString(dtype.value()),
      ".");
  return at::_th_cumsum_out(result, self.toType(result.type().scalarType()), dim);
}

Tensor& cumsum_out(Tensor& result, const Tensor& self, int64_t dim, ScalarType dtype) {
  return at::native::cumsum_out(result, self, dim, optional<ScalarType>(dtype));
}

Tensor& cumsum_out(Tensor& result, const Tensor& self, int64_t dim) {
  return at::native::cumsum_out(result, self, dim, c10::nullopt);
}

static inline Tensor cumprod(const Tensor& self, int64_t dim, optional<ScalarType> dtype) {
  return at::_th_cumprod(integer_upcast(self, dtype), dim);
}

Tensor cumprod(const Tensor& self, int64_t dim, ScalarType dtype) {
  return at::native::cumprod(self, dim, optional<ScalarType>(dtype));
}

Tensor cumprod(const Tensor& self, int64_t dim) {
  return at::native::cumprod(self, dim, c10::nullopt);
}

static inline Tensor& cumprod_out(Tensor& result, const Tensor& self, int64_t dim, optional<ScalarType> dtype) {
  // result type is favored over dtype; check that they match if provided (NumPy doesn't check)
  AT_CHECK(
      !dtype.has_value() || (result.type().scalarType() == dtype.value()),
      "provided dtype must match dtype of result in cumprod. Got ",
      at::toString(result.type().scalarType()),
      " and ",
      at::toString(dtype.value()),
      ".");
  return at::_th_cumprod_out(result, self.toType(result.type().scalarType()), dim);
}

Tensor& cumprod_out(Tensor& result, const Tensor& self, int64_t dim, ScalarType dtype) {
  return at::native::cumprod_out(result, self, dim, optional<ScalarType>(dtype));
}

Tensor& cumprod_out(Tensor& result, const Tensor& self, int64_t dim) {
  return at::native::cumprod_out(result, self, dim, c10::nullopt);
}

// ALL REDUCE #################################################################

static inline Tensor mean(const Tensor &self, optional<ScalarType> dtype) {
  ScalarType scalarType = self.type().scalarType();
  AT_CHECK(
      at::isFloatingType(scalarType),
      "Can only calculate the mean of floating types. Got ",
      at::toString(scalarType),
      " instead.");
  if (self.numel() > 0) {
    Tensor result = at::native::sum(self);
    return result.div_(self.numel());
  } else {
    return self.type().scalarTensor(std::numeric_limits<double>::quiet_NaN());
  }
}

Tensor mean(const Tensor &self, ScalarType dtype) {
  return at::native::mean(self, optional<ScalarType>(dtype));
}

Tensor mean(const Tensor &self) {
  return at::native::mean(self, c10::nullopt);
}

static ScalarType get_dtype(Tensor& result, const Tensor& self, optional<ScalarType> dtype,
                            bool promote_integers=false) {
  if (dtype.has_value()) {
    return dtype.value();
  } else if (result.defined()) {
    return result.type().scalarType();
  }
  ScalarType src_type = self.type().scalarType();
  if (promote_integers && at::isIntegralType(src_type)) {
    return kLong;
  }
  return src_type;
}

static Tensor& sum_out(Tensor& result, const Tensor& self, IntList dim,
                       bool keepdim, optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  auto iter = make_reduction("sum", result, self, dim, keepdim, dtype);
  if (iter->numel() == 0) {
    result.zero_();
  } else {
    sum_stub(iter->device_type(), *iter);
  }
  return result;
}

static Tensor sum(const Tensor& self, IntList dim, bool keepdim, optional<ScalarType> dtype) {
  Tensor result;
  native::sum_out(result, self, dim, keepdim, dtype);
  return result;
}

Tensor sum(const Tensor &self, ScalarType dtype) {
  return at::native::sum(self, {}, false, optional<ScalarType>(dtype));
}

Tensor sum(const Tensor &self) {
  return at::native::sum(self, {}, false, c10::nullopt);
}

static Tensor& prod_out(Tensor& result, const Tensor& self, IntList dim,
                        bool keepdim, optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  auto iter = make_reduction("prod", result, self, dim, keepdim, dtype);
  if (iter->numel() == 0) {
    result.fill_(1);
  } else {
    prod_stub(iter->device_type(), *iter);
  }
  return result;
}

static Tensor prod(const Tensor& self, IntList dim, bool keepdim, optional<ScalarType> dtype) {
  Tensor result;
  native::prod_out(result, self, dim, keepdim, dtype);
  return result;
}

Tensor prod(const Tensor &self, ScalarType dtype) {
  return at::native::prod(self, {}, false, optional<ScalarType>(dtype));
}

Tensor prod(const Tensor &self) {
  return at::native::prod(self, {}, false, c10::nullopt);
}

// \ALL REDUCE ################################################################

// DIM REDUCE #################################################################

static inline Tensor &mean_out(Tensor &result, const Tensor &self, int64_t dim,
                 bool keepdim, optional<ScalarType> dtype) {
  ScalarType scalarType = result.type().scalarType();
  AT_CHECK(
      at::isFloatingType(scalarType),
      "Can only calculate the mean of floating types. Got ",
      at::toString(scalarType),
      " instead.");
  at::native::sum_out(
      result, self.toType(result.type().scalarType()), dim, keepdim);
  if (result.numel() > 0 && self.ndimension() > 0) {
    int64_t numel = self.size(dim);
    if (numel > 0) {
      result.div_(numel);
    } else {
      // NumPy equivalent
      result.fill_(std::numeric_limits<double>::quiet_NaN());
    }
  }
  return result;
}

Tensor& mean_out(Tensor& result, const Tensor& self, int64_t dim, bool keepdim, ScalarType dtype) {
  return at::native::mean_out(
      result, self, dim, keepdim, c10::optional<ScalarType>(dtype));
}
Tensor& mean_out(Tensor& result, const Tensor& self, int64_t dim, bool keepdim) {
  return at::native::mean_out(result, self, dim, keepdim, c10::nullopt);
}

Tensor& mean_out(Tensor& result, const Tensor& self, int64_t dim, ScalarType dtype) {
  return at::native::mean_out(result, self, dim, false, dtype);
}

Tensor& sum_out(Tensor& result, const Tensor& self, IntList dim, bool keepdim, ScalarType dtype) {
  return at::native::sum_out(
      result, self, dim, keepdim, c10::optional<ScalarType>(dtype));
}

Tensor& sum_out(Tensor& result, const Tensor& self, IntList dim, bool keepdim) {
  return at::native::sum_out(result, self, dim, keepdim, c10::nullopt);
}

Tensor& sum_out(Tensor& result, const Tensor& self, IntList dim, ScalarType dtype) {
  return at::native::sum_out(result, self, dim, false, dtype);
}

Tensor& prod_out(Tensor& result, const Tensor& self, int64_t dim, bool keepdim, ScalarType dtype) {
  return at::native::prod_out(
      result, self, dim, keepdim, c10::optional<ScalarType>(dtype));
}

Tensor& prod_out(Tensor& result, const Tensor& self, int64_t dim, bool keepdim) {
  return at::native::prod_out(result, self, dim, keepdim, c10::nullopt);
}

Tensor& prod_out(Tensor& result, const Tensor& self, int64_t dim, ScalarType dtype) {
  return at::native::prod_out(result, self, dim, false, dtype);
}

static inline Tensor mean(const Tensor &self, int64_t dim, bool keepdim, optional<ScalarType> dtype) {
  ScalarType scalarType = self.type().scalarType();
  AT_CHECK(
      at::isFloatingType(scalarType),
      "Can only calculate the mean of floating types. Got ",
      at::toString(scalarType),
      " instead.");
  Tensor result = at::native::sum(self, dim, keepdim);
  if (result.numel() > 0 && self.ndimension() > 0) {
    int64_t numel = self.size(dim);
    if (numel > 0) {
      result.div_(numel);
    } else {
      // NumPy equivalent
      result.fill_(std::numeric_limits<double>::quiet_NaN());
    }
  }
  return result;
}

Tensor mean(const Tensor& self, int64_t dim, bool keepdim, ScalarType dtype) {
  return at::native::mean(self, dim, keepdim, c10::optional<ScalarType>(dtype));
}

Tensor mean(const Tensor& self, int64_t dim, bool keepdim) {
  return at::native::mean(self, dim, keepdim, c10::nullopt);
}

Tensor mean(const Tensor& self, int64_t dim, ScalarType dtype) {
  return at::native::mean(self, dim, false, dtype);
}

Tensor sum(const Tensor& self, IntList dim, bool keepdim, ScalarType dtype) {
  return at::native::sum(self, dim, keepdim, c10::optional<ScalarType>(dtype));
}

Tensor sum(const Tensor& self, IntList dim, bool keepdim) {
  return at::native::sum(self, dim, keepdim, c10::nullopt);
}

Tensor sum(const Tensor& self, IntList dim, ScalarType dtype) {
  return at::native::sum(self, dim, false, dtype);
}

Tensor prod(const Tensor& self, int64_t dim, bool keepdim, ScalarType dtype) {
  return at::native::prod(self, dim, keepdim, c10::optional<ScalarType>(dtype));
}

Tensor prod(const Tensor& self, int64_t dim, bool keepdim) {
  return at::native::prod(self, dim, keepdim, c10::nullopt);
}

Tensor prod(const Tensor& self, int64_t dim, ScalarType dtype) {
  return at::native::prod(self, dim, false, dtype);
}

Tensor& logsumexp_out(Tensor& result, const Tensor &self, int64_t dim_, bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  // can't take max of empty tensor
  if (self.numel() != 0) {
    auto maxes = at::max_values(self, dim, true);
    auto maxes_squeezed = (keepdim ? maxes : maxes.squeeze(dim));
    maxes_squeezed.masked_fill_(maxes_squeezed.abs() == INFINITY, 0);
    at::sum_out(result, at::exp(self - maxes), dim, keepdim);
    result.log_().add_(maxes_squeezed);
  } else {
    at::sum_out(result, at::exp(self), dim, keepdim);
    result.log_();
  }
  return result;
}

Tensor logsumexp(const Tensor &self, int64_t dim_, bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  Tensor result = at::empty({0}, self.options());
  return at::native::logsumexp_out(result, self, dim, keepdim);
}

Tensor& _norm_out_cpu(Tensor& result, const Tensor& self, Scalar p, int64_t dim_, bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  if (_dimreduce_return_trivial(result, self, 0, dim, keepdim))
    return result;
  if (self.is_contiguous() && result.is_contiguous()) {
    _dimreduce_setup(result, self, dim);
    norm_kernel(kCPU, result, self, p, dim);
    if (!keepdim) {
      result.squeeze_(dim);
    }
    return result;
  } else {
    return at::_th_norm_out(result, self, p, dim, keepdim);
  }
}

Tensor& norm_out(Tensor &result, const Tensor &self, Scalar p, int64_t dim, bool keepdim) {
  AT_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "norm only supports CPU AND CUDA backend, got: ", at::toString(self.type().backend()));
  AT_CHECK(at::isFloatingType(self.type().scalarType()), "norm only supports floating-point dtypes");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial(result, self, 0, dim, keepdim)) {
    return result;
  } else {
    if (self.is_cuda()) {
      return at::_th_norm_out(result, self, p, dim, keepdim);
    } else {
      return _norm_out_cpu(result, self, p, dim, keepdim);
    }
  }
}

Tensor _norm(const Tensor &self, Scalar p) {
  if (self.is_sparse()) {
    return at::native_norm(self, p);
  } else {
    AT_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
             "norm only supports CPU AND CUDA backend, got: ", at::toString(self.type().backend()));
    AT_CHECK(at::isFloatingType(self.type().scalarType()), "norm only supports floating-point dtypes");
    if (self.is_cuda()) {
      return at::_th_norm(self, p);
    } else {
      if (self.is_contiguous()) {
        Tensor result = CPU(kFloat).scalarTensor(0).toType(self.type());
        norm_kernel(kCPU, result, self, p, c10::nullopt);
        return result;
      } else {
        return at::_th_norm(self, p);
      }
    }
  }
}

Tensor norm(const Tensor& self, Scalar p, int64_t dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::norm_out(result, self, p, dim, keepdim);
}

Tensor norm(const Tensor& self, Scalar p) {
  return at::native::_norm(self, p);
}

Tensor all(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::all_out(result, self, dim, keepdim);
}

Tensor &all_out(Tensor &result, const Tensor &self, int64_t dim, bool keepdim) {
  AT_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "all only supports CPU AND CUDA backend, got: ", at::toString(self.type().backend()));
  AT_CHECK(self.type().scalarType() == at::ScalarType::Byte, "all only supports torch.uint8 dtype");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial(result, self, 1, dim, keepdim)) {
    return result;
  } else {
    return at::_th_all_out(result, self, dim, keepdim);
  }
}

Tensor any(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::any_out(result, self, dim, keepdim);
}

Tensor &any_out(Tensor &result, const Tensor &self, int64_t dim, bool keepdim) {
  AT_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "any only supports CPU AND CUDA backend, got: ", at::toString(self.type().backend()));
  AT_CHECK(self.type().scalarType() == at::ScalarType::Byte, "any only supports torch.uint8 dtype");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial(result, self, 0, dim, keepdim)) {
    return result;
  } else {
    return at::_th_any_out(result, self, dim, keepdim);
  }
}

Tensor var(const Tensor& self, bool unbiased) {
  AT_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "var only supports CPU AND CUDA backend, got: ", at::toString(self.type().backend()));
  AT_CHECK(at::isFloatingType(self.type().scalarType()), "var only supports floating-point dtypes");
  auto trivial_return = _allreduce_return_trivial(self, std::numeric_limits<double>::quiet_NaN());
  return trivial_return.has_value() ? trivial_return.value() : at::_th_var(self, unbiased);
}

Tensor var(const Tensor& self, int64_t dim, bool unbiased, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::var_out(result, self, dim, unbiased, keepdim);
}

Tensor &var_out(Tensor &result, const Tensor &self, int64_t dim, bool unbiased, bool keepdim) {
  AT_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "var only supports CPU AND CUDA backend, got: ", at::toString(self.type().backend()));
  AT_CHECK(at::isFloatingType(self.type().scalarType()), "var only supports floating-point dtypes");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial(result, self, std::numeric_limits<double>::quiet_NaN(), dim, keepdim)) {
    return result;
  } else {
    return at::_th_var_out(result, self, dim, unbiased, keepdim);
  }
}

Tensor std(const Tensor& self, bool unbiased) {
  AT_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "std only supports CPU AND CUDA backend, got: ", at::toString(self.type().backend()));
  AT_CHECK(at::isFloatingType(self.type().scalarType()), "std only supports floating-point dtypes");
  auto trivial_return = _allreduce_return_trivial(self, std::numeric_limits<double>::quiet_NaN());
  return trivial_return.has_value() ? trivial_return.value() : at::_th_std(self, unbiased);
}

Tensor std(const Tensor& self, int64_t dim, bool unbiased, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::std_out(result, self, dim, unbiased, keepdim);
}

Tensor &std_out(Tensor &result, const Tensor &self, int64_t dim, bool unbiased, bool keepdim) {
  AT_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "std only supports CPU AND CUDA backend, got: ", at::toString(self.type().backend()));
  AT_CHECK(at::isFloatingType(self.type().scalarType()), "std only supports floating-point dtypes");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial(result, self, std::numeric_limits<double>::quiet_NaN(), dim, keepdim)) {
    return result;
  } else {
    return at::_th_std_out(result, self, dim, unbiased, keepdim);
  }
}

}} // namespace at::native
