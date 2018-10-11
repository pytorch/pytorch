#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/WrapDimUtilsMulti.h"
#include "ReduceOpsUtils.h"
#include "cpu/ReduceOpsKernel.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>
#include <map>

namespace at {
namespace native {

DEFINE_DISPATCH(sum_kernel);
DEFINE_DISPATCH(prod_kernel);
DEFINE_DISPATCH(norm_kernel);

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
  return at::_cumsum_out(result, self.toType(result.type().scalarType()), dim);
}

Tensor& cumsum_out(Tensor& result, const Tensor& self, int64_t dim, ScalarType dtype) {
  return at::native::cumsum_out(result, self, dim, optional<ScalarType>(dtype));
}

Tensor& cumsum_out(Tensor& result, const Tensor& self, int64_t dim) {
  return at::native::cumsum_out(result, self, dim, c10::nullopt);
}

static inline Tensor cumprod(const Tensor& self, int64_t dim, optional<ScalarType> dtype) {
  return at::_cumprod(integer_upcast(self, dtype), dim);
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
  return at::_cumprod_out(result, self.toType(result.type().scalarType()), dim);
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

static inline Tensor sum(const Tensor &self, optional<ScalarType> dtype) {
  return at::_sum(integer_upcast(self, dtype));
}

Tensor sum(const Tensor &self, ScalarType dtype) {
  return at::native::sum(self, optional<ScalarType>(dtype));
}

Tensor sum(const Tensor &self) {
  return at::native::sum(self, c10::nullopt);
}

Tensor _sum_cpu(const Tensor& self) {
  if (self.is_contiguous()) {
    Tensor result = at::empty({}, self.type());
    sum_kernel(kCPU, result, self, c10::nullopt);
    return result;
  }
  return at::_sumall(self);
}

static inline Tensor prod(const Tensor &self, optional<ScalarType> dtype) {
  return at::_prod(integer_upcast(self, dtype));
}

Tensor prod(const Tensor &self, ScalarType dtype) {
  return at::native::prod(self, optional<ScalarType>(dtype));
}

Tensor prod(const Tensor &self) {
  return at::native::prod(self, c10::nullopt);
}

Tensor _prod_cpu(const Tensor &self) {
  if (self.is_contiguous()) {
    Tensor result = at::empty({}, self.type());
    prod_kernel(kCPU, result, self, c10::nullopt);
    return result;
  }
  return at::_prodall(self);
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
  return at::native::sum_out(
      result, self, dim, keepdim, c10::optional<ScalarType>(dtype));
}
Tensor& sum_out(Tensor& result, const Tensor& self, IntList dim, bool keepdim) {
  return at::native::sum_out(result, self, dim, keepdim, c10::nullopt);
}

Tensor& sum_out(Tensor& result, const Tensor& self, IntList dim, ScalarType dtype) {
  return at::native::sum_out(result, self, dim, false, dtype);
}

Tensor &_sum_out_cpu(Tensor &result, const Tensor &self, int64_t dim_,
                     bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  if (_dimreduce_return_trivial(result, self, 0, dim, keepdim))
    return result;
  if (self.is_contiguous() && result.is_contiguous()) {
    _dimreduce_setup(result, self, dim);
    sum_kernel(kCPU, result, self, dim);
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
  return at::native::prod_out(
      result, self, dim, keepdim, c10::optional<ScalarType>(dtype));
}
Tensor& prod_out(Tensor& result, const Tensor& self, int64_t dim, bool keepdim) {
  return at::native::prod_out(result, self, dim, keepdim, c10::nullopt);
}

Tensor& prod_out(Tensor& result, const Tensor& self, int64_t dim, ScalarType dtype) {
  return at::native::prod_out(result, self, dim, false, dtype);
}

Tensor &_prod_out_cpu(Tensor &result, const Tensor &self, int64_t dim_,
                      bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  if (_dimreduce_return_trivial(result, self, 1, dim, keepdim))
    return result;
  if (self.is_contiguous() && result.is_contiguous()) {
    _dimreduce_setup(result, self, dim);
    prod_kernel(kCPU, result, self, dim);
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

static inline Tensor sum(const Tensor &self, IntList dim_, bool keepdim, optional<ScalarType> dtype) {
  return at::_sum(integer_upcast(self, dtype), dim_, keepdim);
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

Tensor _sum(const Tensor &self, int64_t dim_, bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  Tensor result = at::empty({0}, self.options());
  return at::_sum_out(result, self, dim, keepdim);
}

static inline Tensor prod(const Tensor &self, int64_t dim_, bool keepdim, optional<ScalarType> dtype) {
  return at::_prod(integer_upcast(self, dtype), dim_, keepdim);
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

Tensor _prod(const Tensor &self, int64_t dim_, bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  Tensor result = at::empty({0}, self.options());
  return at::_prod_out(result, self, dim, keepdim);
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

// \DIM REDUCE ################################################################

// MULTI DIM REDUCE ###########################################################

// NB: this applies two optimizations:
//   1. Reducing the dimensions in the order of decreasing size, so that the
//      larger dimensions are dealt earlier and we can work with less elements
//      overall.
//      E.g., reducing tensor of shape [1, 10, 200] over dimemsions {0, 1, 2}.
//            If we reduce in the order of [0, 1, 2], the input and output
//            shapes of iterations are:
//                it 0:  [1, 10, 200] (2000 elem) => [10, 200] (2000 elem)
//                it 1:     [10, 200] (2000 elem) =>     [200] ( 200 elem)
//                it 2:         [200] ( 200 elem) =>     [  1] (   1 elem)
//              Since we need to iterate through all input elements at each
//              iteration, total number of elements traversed is 4200.
//            If we reduce in the order of [2, 1, 0], i.e., with decreasing
//            size, the input and output shapes of iterations are:
//                it 0:  [1, 10, 200] (2000 elem) => [1, 10] (10 elem)
//                it 1:      [1,  10] (  10 elem) =>    [ 1] ( 1 elem)
//                it 2:           [1] (   1 elem) =>    [ 1] ( 1 elem)
//              Total number of elements traversed is 2011, much less than 4200.
//   2. Preallocated buffer.
//      Utilizing the `_out` variant, instead of allocating new output tensors
//      at each iteration, we can use a preallocated buffer. Since output numel
//      in each iteration is decreasing, we can reuse the buffer throughout the
//      loop.
//      Note that we need two buffers, one containing the input, i.e., output
//      from the previous iteration, and one containing the output for this
//      iteration.
//      The largest output size is the output size of the first iteration. After
//      that the largest size we need is the output size  of the second
//      iteration.
//      So we allocate
//        1. a region of size `input.numel() / input.size(reduced_dims[0])`, and
//        2. a region of size `input.numel() / (input.size(reduced_dims[0]) * input.size(reduced_dims[1]))`.
//      These two regions are allocated together as a contiguous flattened
//      buffer tensor, with a variable `offset` indicating the starting position
//      of the output region for the current iteration.
//      E.g., reducing tensor of shape [4, 3, 2] over dimemsions {0, 1, 2}.
//            Say we reduce in the order of [0, 1, 2].
//            The first buffer with has size `4 * 3 * 2 / 4 = 6`.
//            The second buffer with has size `4 * 3 * 2 / (4 * 3) = 2`.
//            So we allocate a tensor of size `6 + 2 = 8`:
//              buffer: [ _, _, _, _, _, _, _, _]
//      buffer region 1-->^^^^^^^^^^^^^^^^  ^^^^<--buffer region 2
//            1st iteration:
//              (before reduction)
//                input:         self (or input)
//                input shape:   [ 4, 3, 2]
//                output shape:  [ 3, 2]
//                buffer:        [ _, _, _, _, _, _, _, _]
//                offset:          ^--beginning of 1st buffer region, i.e., the
//                                    starting output location of 1st iteration.
//              (after reduction)
//                buffer:        [ {output of 1st it}, _, _]
//
//            2nd iteration:
//              (before reduction)
//                input:         output of 1st it
//                input shape:   [ 3, 2]
//                output shape:  [ 2]
//                buffer:        [ {output of 1st it}, _, _]
//                offset:                              ^--beginning of 2nd
//                                                      buffer region. We can't
//                                                      overwrite the 1st buffer
//                                                      as it contains input to
//                                                      reduction of this it.
//              (after reduction)
//                buffer:        [ {output of 1st it}, {output of 2nd it}]
//
//            3rd iteration:
//              (before reduction)
//                input:         output of 2nd it
//                input shape:   [ 2]
//                output shape:  [ 1]
//                buffer:        [ {output of 1st it}, {output of 2nd it}]
//                offset:          ^--beginning of 1st buffer region. We can
//                                  safely overwrite now.
//              (after reduction)
//                buffer:        [ {output of 3rd it}, {output of 2nd it}]
//            Return {output of 3rd it}.
//
// TODO: If two or more reduced dimensions are contiguous, reduce as if they are
//       a large dimension.
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
  // `reduced_numel` and `reduced_size` will be updated in the loop.
  // Before that, they are just size and numel.
  int64_t reduced_numel = self.numel();
  auto reduced_size = self.sizes().vec();
  auto dims = dims_.vec();
  maybe_wrap_dims(dims, ndims);
  // Sort the reduced dimensions so that we reduce the larger dimensions first.
  std::sort(dims.begin(), dims.end(),
        [&](int64_t i, int64_t j){ return reduced_size[i] > reduced_size[j]; });
  // Calculate 1st buffer region size
  int64_t max_reduced_numel = reduced_numel / reduced_size[dims[0]];
  int64_t buffer_size = max_reduced_numel + max_reduced_numel / reduced_size[dims[1]];
  // We separate `buffer` into two regions, one starting at 0, and another
  // starting at max_reduced_numel. These two regions are used alternatively as
  // the output of a `reduce_1` along a particular dimension. `offset` will
  // indicate which region we should use next.
  // Have keepdim=true when reducing. We will squeeze later.
  auto buffer = at::empty({buffer_size}, self.options());
  int64_t offset = 0;
  Tensor t = self;
  for (auto& dim : dims) {
    reduced_numel /= reduced_size[dim];
    reduced_size[dim] = 1;
    auto res = buffer.narrow(0, offset, reduced_numel).view(reduced_size);
    t = reduce_1_out(res, t, dim, true);
    // switch to other buffer region
    // this alternatively changes `offset` between 0 and max_reduced_numel
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

// See comments above reduce_multi_associative for details.
template <Tensor (reduce_1)(const Tensor &, int64_t, bool),
    Tensor& (reduce_1_out)(Tensor& result, const Tensor &, int64_t, bool)>
inline Tensor& reduce_multi_associative_out(Tensor &result, const Tensor &self, IntList dims_, bool keepdim) {
  if (dims_.size() == 1) {
    return reduce_1_out(result, self, dims_[0], keepdim);
  }
  if (dims_.size() == 0) {
    // reduce_out should be clone_out with empty dims_
    return result.resize_as_(self).copy_(self);
  }
  int64_t ndims = self.dim();
  // `reduced_numel` and `reduced_size` will be updated in the loop.
  // Before that, they are just size and numel.
  int64_t reduced_numel = self.numel();
  auto reduced_size = self.sizes().vec();
  auto dims = dims_.vec();
  maybe_wrap_dims(dims, ndims);
  // Sort the reduced dimensions so that we reduce the largest dimension first.
  std::sort(dims.begin(), dims.end(),
        [&](int64_t i, int64_t j){ return reduced_size[i] > reduced_size[j]; });
  // Calculate 1st buffer region size
  int64_t max_reduced_numel = reduced_numel / reduced_size[dims[0]];
  int64_t buffer_size = max_reduced_numel + max_reduced_numel / reduced_size[dims[1]];
  // We separate `buffer` into two regions, one starting at 0, and another
  // starting at max_reduced_numel. These two regions are used alternatively as
  // the output of a `reduce_1` along a particular dimension. `offset` will
  // indicate which region we should use next.
  // Have keepdim=true when reducing. We will squeeze later.
  auto buffer = at::empty({buffer_size}, self.options());
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
    // switch to other buffer region
    // this alternatively changes `offset` between 0 and max_reduced_numel
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
  if (self.type().is_sparse()) {
    return at::native_norm(self, p);
  } else {
    AT_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
             "norm only supports CPU AND CUDA backend, got: ", at::toString(self.type().backend()));
    AT_CHECK(at::isFloatingType(self.type().scalarType()), "norm only supports floating-point dtypes");
    if (self.is_cuda()) {
      return at::th_norm(self, p);
    } else {
      if (self.is_contiguous()) {
        Tensor result = CPU(kFloat).scalarTensor(0).toType(self.type());
        norm_kernel(kCPU, result, self, p, c10::nullopt);
        return result;
      } else {
        return at::th_norm(self, p);
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
