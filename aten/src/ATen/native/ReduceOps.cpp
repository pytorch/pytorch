#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/WrapDimUtilsMulti.h"
#include "cpu/ReduceOpsKernel.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include <map>

namespace at {
namespace native {

static inline Tensor integer_upcast(const Tensor& self, optional<ScalarType> dtype) {
  ScalarType scalarType = self.type().scalarType();
  ScalarType upcast_scalarType = dtype.value_or(at::isIntegralType(scalarType) ? ScalarType::Long : scalarType);
  return self.toType(upcast_scalarType);
}

static inline Tensor cumsum(const Tensor& self, int64_t dim, optional<ScalarType> dtype) {
  return at::_cumsum(integer_upcast(self, dtype), dim);
}

Tensor cumsum(const Tensor& self, int64_t dim, ScalarType dtype) {
  return at::native::cumsum(self, dim, optional<ScalarType>(dtype));
}

Tensor cumsum(const Tensor& self, int64_t dim) {
  return at::native::cumsum(self, dim, nullopt);
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
  return at::_cumsum_out(result, self.toType(result.type().scalarType()), dim);
}

Tensor& cumsum_out(Tensor& result, const Tensor& self, int64_t dim, ScalarType dtype) {
  return at::native::cumsum_out(result, self, dim, optional<ScalarType>(dtype));
}

Tensor& cumsum_out(Tensor& result, const Tensor& self, int64_t dim) {
  return at::native::cumsum_out(result, self, dim, nullopt);
}

static inline Tensor cumprod(const Tensor& self, int64_t dim, optional<ScalarType> dtype) {
  return at::_cumprod(integer_upcast(self, dtype), dim);
}

Tensor cumprod(const Tensor& self, int64_t dim, ScalarType dtype) {
  return at::native::cumprod(self, dim, optional<ScalarType>(dtype));
}

Tensor cumprod(const Tensor& self, int64_t dim) {
  return at::native::cumprod(self, dim, nullopt);
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
  return at::_cumprod_out(result, self.toType(result.type().scalarType()), dim);
}

Tensor& cumprod_out(Tensor& result, const Tensor& self, int64_t dim, ScalarType dtype) {
  return at::native::cumprod_out(result, self, dim, optional<ScalarType>(dtype));
}

Tensor& cumprod_out(Tensor& result, const Tensor& self, int64_t dim) {
  return at::native::cumprod_out(result, self, dim, nullopt);
}

// ALL REDUCE #################################################################

static inline Tensor mean(const Tensor &self, optional<ScalarType> dtype) {
  ScalarType scalarType = self.type().scalarType();
  AT_CHECK(
      at::isFloatingType(scalarType),
      "Can only calculate the mean of floating types. Got ",
      at::toString(scalarType),
      " instead.");
  Tensor result = at::native::sum(self);
  if (self.numel() > 0)
    result.div_(self.numel());
  return result;
}

Tensor mean(const Tensor &self, ScalarType dtype) {
  return at::native::mean(self, optional<ScalarType>(dtype));
}

Tensor mean(const Tensor &self) {
  return at::native::mean(self, nullopt);
}

static inline Tensor sum(const Tensor &self, optional<ScalarType> dtype) {
  return at::_sum(integer_upcast(self, dtype));
}

Tensor sum(const Tensor &self, ScalarType dtype) {
  return at::native::sum(self, optional<ScalarType>(dtype));
}

Tensor sum(const Tensor &self) {
  return at::native::sum(self, nullopt);
}

Tensor _sum_cpu(const Tensor& self) {
  if (self.is_contiguous()) {
    Tensor result = at::empty({}, self.type());
    sum_kernel(result, self, at::nullopt);
    return result;
  }
  return self._sumall();
}

static inline Tensor prod(const Tensor &self, optional<ScalarType> dtype) {
  return at::_prod(integer_upcast(self, dtype));
}

Tensor prod(const Tensor &self, ScalarType dtype) {
  return at::native::prod(self, optional<ScalarType>(dtype));
}

Tensor prod(const Tensor &self) {
  return at::native::prod(self, nullopt);
}

Tensor _prod_cpu(const Tensor &self) {
  if (self.is_contiguous()) {
    Tensor result = at::empty({}, self.type());
    prod_kernel(result, self, at::nullopt);
    return result;
  }
  return self._prodall();
}

// \ALL REDUCE ################################################################

// DIM REDUCE #################################################################

static bool _dimreduce_return_trivial(Tensor &result, const Tensor &self,
                                      int64_t ident) {
  if (self.numel() == 1 && self.ndimension() == 0) {
    result.resize_({});
    result.fill_(self);
    return true;
  }
  // Return identity
  if (self.numel() == 0 && self.ndimension() == 1) {
    result.resize_({0});
    result.fill_(ident);
    return true;
  }
  return false;
}

static Tensor &_dimreduce_setup(Tensor &result, const Tensor &self,
                                int64_t dim) {
  IntList self_sizes = self.sizes();
  std::vector<int64_t> result_sizes;
  result_sizes.insert(result_sizes.end(), self_sizes.begin(), self_sizes.end());
  result_sizes[dim] = 1;
  result.resize_(result_sizes);
  return result;
}

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
    result.div_(numel);
  }
  return result;
}

Tensor& mean_out(Tensor& result, const Tensor& self, int64_t dim, bool keepdim, ScalarType dtype) {
  return at::native::mean_out(result, self, dim, keepdim, at::optional<ScalarType>(dtype));
}
Tensor& mean_out(Tensor& result, const Tensor& self, int64_t dim, bool keepdim) {
  return at::native::mean_out(result, self, dim, keepdim, nullopt);
}

Tensor& mean_out(Tensor& result, const Tensor& self, int64_t dim, ScalarType dtype) {
  return at::native::mean_out(result, self, dim, false, dtype);
}

static inline Tensor &sum_out(Tensor &result, const Tensor &self, IntList dim,
                 bool keepdim, optional<ScalarType> dtype) {
  // result type is favored over dtype; check that they match if provided (NumPy doesn't check)
  AT_CHECK(
      !dtype.has_value() || (result.type().scalarType() == dtype.value()),
      "provided dtype must match dtype of result in sum. Got ",
      at::toString(result.type().scalarType()),
      " and ",
      at::toString(dtype.value()),
      ".");
  return at::_sum_out(result, self.toType(result.type().scalarType()), dim, keepdim);
}

Tensor& sum_out(Tensor& result, const Tensor& self, IntList dim, bool keepdim, ScalarType dtype) {
  return at::native::sum_out(result, self, dim, keepdim, at::optional<ScalarType>(dtype));
}
Tensor& sum_out(Tensor& result, const Tensor& self, IntList dim, bool keepdim) {
  return at::native::sum_out(result, self, dim, keepdim, nullopt);
}

Tensor& sum_out(Tensor& result, const Tensor& self, IntList dim, ScalarType dtype) {
  return at::native::sum_out(result, self, dim, false, dtype);
}

Tensor &_sum_out_cpu(Tensor &result, const Tensor &self, int64_t dim_,
                     bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  if (_dimreduce_return_trivial(result, self, 0))
    return result;
  if (self.is_contiguous() && result.is_contiguous()) {
    _dimreduce_setup(result, self, dim);
    sum_kernel(result, self, dim);
    if (!keepdim) result.squeeze_(dim);
    return result;
  }
  return at::_th_sum_out(result, self, dim, keepdim);
}

static inline Tensor &prod_out(Tensor &result, const Tensor &self, int64_t dim,
                 bool keepdim, optional<ScalarType> dtype) {
  // result type is favored over dtype; check that they match if provided (NumPy doesn't check)
  AT_CHECK(
      !dtype.has_value() || (result.type().scalarType() == dtype.value()),
      "provided dtype must match dtype of result in prod. Got ",
      at::toString(result.type().scalarType()),
      " and ",
      at::toString(dtype.value()),
      ".");
  return at::_prod_out(result, self.toType(result.type().scalarType()), dim, keepdim);
}

Tensor& prod_out(Tensor& result, const Tensor& self, int64_t dim, bool keepdim, ScalarType dtype) {
  return at::native::prod_out(result, self, dim, keepdim, at::optional<ScalarType>(dtype));
}
Tensor& prod_out(Tensor& result, const Tensor& self, int64_t dim, bool keepdim) {
  return at::native::prod_out(result, self, dim, keepdim, nullopt);
}

Tensor& prod_out(Tensor& result, const Tensor& self, int64_t dim, ScalarType dtype) {
  return at::native::prod_out(result, self, dim, false, dtype);
}

Tensor &_prod_out_cpu(Tensor &result, const Tensor &self, int64_t dim_,
                      bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  if (_dimreduce_return_trivial(result, self, 1))
    return result;
  if (self.is_contiguous() && result.is_contiguous()) {
    _dimreduce_setup(result, self, dim);
    prod_kernel(result, self, dim);
    if (!keepdim) result.squeeze_(dim);
    return result;
  }
  return at::_th_prod_out(result, self, dim, keepdim);
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
    result.div_(numel);
  }
  return result;
}

Tensor mean(const Tensor& self, int64_t dim, bool keepdim, ScalarType dtype) {
  return at::native::mean(self, dim, keepdim, at::optional<ScalarType>(dtype));
}

Tensor mean(const Tensor& self, int64_t dim, bool keepdim) {
  return at::native::mean(self, dim, keepdim, nullopt);
}

Tensor mean(const Tensor& self, int64_t dim, ScalarType dtype) {
  return at::native::mean(self, dim, false, dtype);
}

static inline Tensor sum(const Tensor &self, IntList dim_, bool keepdim, optional<ScalarType> dtype) {
  return at::_sum(integer_upcast(self, dtype), dim_, keepdim);
}

Tensor sum(const Tensor& self, IntList dim, bool keepdim, ScalarType dtype) {
  return at::native::sum(self, dim, keepdim, at::optional<ScalarType>(dtype));
}

Tensor sum(const Tensor& self, IntList dim, bool keepdim) {
  return at::native::sum(self, dim, keepdim, nullopt);
}

Tensor sum(const Tensor& self, IntList dim, ScalarType dtype) {
  return at::native::sum(self, dim, false, dtype);
}

Tensor _sum(const Tensor &self, int64_t dim_, bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  Tensor result = self.type().tensor();
  return at::_sum_out(result, self, dim, keepdim);
}

static inline Tensor prod(const Tensor &self, int64_t dim_, bool keepdim, optional<ScalarType> dtype) {
  return at::_prod(integer_upcast(self, dtype), dim_, keepdim);
}

Tensor prod(const Tensor& self, int64_t dim, bool keepdim, ScalarType dtype) {
  return at::native::prod(self, dim, keepdim, at::optional<ScalarType>(dtype));
}

Tensor prod(const Tensor& self, int64_t dim, bool keepdim) {
  return at::native::prod(self, dim, keepdim, nullopt);
}

Tensor prod(const Tensor& self, int64_t dim, ScalarType dtype) {
  return at::native::prod(self, dim, false, dtype);
}

Tensor _prod(const Tensor &self, int64_t dim_, bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  Tensor result = self.type().tensor();
  return at::_prod_out(result, self, dim, keepdim);
}

Tensor& logsumexp_out(Tensor& result, const Tensor &self, int64_t dim_, bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  auto maxes = at::max_values(self, dim, true);
  result = at::where((maxes == INFINITY).__or__(maxes == -INFINITY),
		     maxes,
		     maxes + at::log(at::sum(at::exp(self - maxes), dim, true)));
  if (! keepdim)
    result.squeeze_(dim);
  return result;
}

Tensor logsumexp(const Tensor &self, int64_t dim_, bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  Tensor result = self.type().tensor();
  return at::native::logsumexp_out(result, self, dim, keepdim);
}

// \DIM REDUCE ################################################################

// MULTI DIM REDUCE ###########################################################

template <Tensor (reduce_1)(const Tensor &, int64_t, bool),
    Tensor& (reduce_1_out)(Tensor& result, const Tensor &, int64_t, bool)>
inline Tensor reduce_multi_associative(const Tensor &self, IntList dims_, bool keepdim) {
  if (dims_.size() == 1) {
    return reduce_1(self, dims_[0], keepdim);
  }
  if (dims_.size() == 0) {
    return self;
  }
  int64_t ndims = self.dim();
  auto reduced_size = self.sizes().vec();
  auto dims = dims_.vec();
  maybe_wrap_dims(dims, ndims);
  // Sort the reduced dimensions so that we reduce the largest dimension first.
  std::sort(dims.begin(), dims.end(),
        [&](int64_t i, int64_t j){ return reduced_size[i] > reduced_size[j]; });
  int64_t reduced_numel = self.numel();
  int64_t max_reduced_numel = reduced_numel / reduced_size[dims[0]];
  int64_t buffer_size = max_reduced_numel + max_reduced_numel / reduced_size[dims[1]];
  // We separate `buffer` into two regions, one starting at 0, and another
  // starting at max_reduced_numel. These two regions are used alternatively as
  // the output of a `reduce_1` along a particular dimension. `offset` will
  // indicate which region we should use next.
  // Have keepdim=true when reducing. We will squeeze later.
  auto buffer = at::empty({buffer_size}, self.type());
  int64_t offset = 0;
  Tensor t = self;
  for (auto& dim : dims) {
    reduced_numel /= reduced_size[dim];
    reduced_size[dim] = 1;
    auto res = buffer.narrow(0, offset, reduced_numel).view(reduced_size);
    t = reduce_1_out(res, t, dim, true);
    offset = max_reduced_numel - offset;
  }
  // squeeze if needed
  if (!keepdim) {
    std::vector<int64_t> squeezed_shape;
    squeezed_shape.reserve(ndims - dims.size());
    auto reduce_dims = dim_list_to_bitset(dims_, ndims);
    for (int64_t dim = 0; dim < ndims; dim++) {
      if (!reduce_dims[dim]) {
        squeezed_shape.emplace_back(reduced_size[dim]);
      }
    }
    return t.view(squeezed_shape);
  }
  return t;
}

template <Tensor (reduce_1)(const Tensor &, int64_t, bool),
    Tensor& (reduce_1_out)(Tensor& result, const Tensor &, int64_t, bool)>
inline Tensor& reduce_multi_associative_out(Tensor &result, const Tensor &self, IntList dims_, bool keepdim) {
  if (dims_.size() == 1) {
    return reduce_1_out(result, self, dims_[0], keepdim);
  }
  if (dims_.size() == 0) {
    return result;
  }
  int64_t ndims = self.dim();
  auto reduced_size = self.sizes().vec();
  auto dims = dims_.vec();
  maybe_wrap_dims(dims, ndims);
  // Sort the reduced dimensions so that we reduce the largest dimension first.
  std::sort(dims.begin(), dims.end(),
        [&](int64_t i, int64_t j){ return reduced_size[i] > reduced_size[j]; });
  int64_t reduced_numel = self.numel();
  int64_t max_reduced_numel = reduced_numel / reduced_size[dims[0]];
  int64_t buffer_size = max_reduced_numel + max_reduced_numel / reduced_size[dims[1]];
  // We separate `buffer` into two regions, one starting at 0, and another
  // starting at max_reduced_numel. These two regions are used alternatively as
  // the output of a `reduce_1` along a particular dimension. `offset` will
  // indicate which region we should use next.
  // Have keepdim=true when reducing. We will squeeze later.
  auto buffer = at::empty({buffer_size}, self.type());
  int64_t offset = 0;
  Tensor t = self;
  int64_t last_reduction = dims.size() - 1;
  int64_t num_reduction = 0;
  for (auto& dim : dims) {
    reduced_numel /= reduced_size[dim];
    reduced_size[dim] = 1;
    auto res = buffer.narrow(0, offset, reduced_numel).view(reduced_size);
    if (num_reduction < last_reduction) {
      t = reduce_1_out(res, t, dim, true);
    } else {
      reduce_1_out(result, t, dim, true);
    }
    offset = max_reduced_numel - offset;
    num_reduction++;
  }
  // squeeze if needed (use in-place squeeze_)
  if (!keepdim) {
    auto reduce_dims = dim_list_to_bitset(dims_, ndims);
    for (int64_t dim = ndims - 1; dim >= 0; dim--) {
      if (reduce_dims[dim]) {
        result.squeeze_(dim);
      }
    }
  }
  return result;
}

Tensor& _sum_out(Tensor &result, const Tensor &self, int64_t dim, bool keepdim) {
  if (self.is_cuda()) {
    return at::_sum_cuda_out(result, self, dim, keepdim);
  } else {
    return _sum_out_cpu(result, self, dim, keepdim);
  }
}

Tensor _sum(const Tensor &self, IntList dims, bool keepdim) {
  return reduce_multi_associative<_sum, _sum_out>(self, dims, keepdim);
}

Tensor& _sum_out(Tensor &result, const Tensor &self, IntList dims, bool keepdim)
{
  return reduce_multi_associative_out<_sum, _sum_out>(result, self, dims, keepdim);
}

}} // namespace at::native
