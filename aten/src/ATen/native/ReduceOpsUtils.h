#pragma once

#include <limits>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/NonEmptyUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <c10/core/ScalarType.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/scalar_tensor.h>
#endif

namespace at::native {

// Maximum and minimum possible scalar values, including infinities
template <typename scalar_t>
constexpr scalar_t upper_bound() {
  using lim = std::numeric_limits<scalar_t>;
  return lim::has_infinity ? lim::infinity() : lim::max();
}

template <typename scalar_t>
constexpr scalar_t lower_bound() {
  using lim = std::numeric_limits<scalar_t>;
  return lim::has_infinity ? -lim::infinity() : lim::lowest();
}

inline Tensor restride_dim(
  const Tensor& src, int64_t dim,
  IntArrayRef replacement_shape
) {
  auto strides = ensure_nonempty_vec(src.strides().vec());
  strides[dim] = 0;
  return src.as_strided(replacement_shape, strides);
}

inline void _dimreduce_setup(const Tensor &result, const Tensor &self,
                                int64_t dim) {
  IntArrayRef self_sizes = self.sizes();
  std::vector<int64_t> result_sizes;
  result_sizes.insert(result_sizes.end(), self_sizes.begin(), self_sizes.end());
  result_sizes[dim] = 1;
  result.resize_(result_sizes);
}

inline bool _dimreduce_return_trivial(const Tensor &result, const Tensor &self,
                                      const Scalar& ident, int64_t dim, bool keepdim) {
  if (self.numel() == 1 && self.ndimension() == 0) {
    result.resize_({});
    result.fill_(self);
    return true;
  }
  // Return identity
  if (self.numel() == 0) {
    _dimreduce_setup(result, self, dim);
    result.fill_(ident);
    if (!keepdim) result.squeeze_(dim);
    return true;
  }
  return false;
}

inline bool _dimreduce_return_trivial_no_ident(Tensor &result, const Tensor &self,
                                               int64_t /*dim*/, bool /*keepdim*/, const char* /*fn_name*/) {
  if (self.numel() == 1 && self.ndimension() == 0) {
    result.resize_({});
    result.fill_(self);
    return true;
  }

  return false;
}

inline std::optional<Tensor> _allreduce_return_trivial(
    const Tensor& self,
    const Scalar& ident) {
  // Return identity
  if (self.numel() == 0) {
    return at::scalar_tensor(ident, self.options());
  }
  return std::nullopt;
}

#define OPTION_TYPE_EQUALITY_CHECK(option, out, self) \
{ \
  TORCH_CHECK(\
    out.option() == self.option(),\
    "expected ", #option, " ",\
    self.option(),\
    " but found ", out.option())\
}

inline void check_scalar_type_device_layout_equal(const Tensor& out, const Tensor& self) {
  OPTION_TYPE_EQUALITY_CHECK(scalar_type, out, self);
  OPTION_TYPE_EQUALITY_CHECK(device, out.options(), self.options());
  OPTION_TYPE_EQUALITY_CHECK(layout, out.options(), self.options());
}

inline Tensor integer_upcast(const Tensor& self, std::optional<ScalarType> dtype) {
  ScalarType scalarType = self.scalar_type();
  TORCH_CHECK(!isBarebonesUnsignedType(scalarType), "integer upcasting for uint16, uint32 and uint64 is not currently implemented");
  ScalarType upcast_scalarType = dtype.value_or(at::isIntegralType(scalarType, /*includeBool=*/true) ? ScalarType::Long : scalarType);
  return self.toType(upcast_scalarType);
}

using DimMask = TensorIterator::DimMask;

inline DimVector make_dim_vector(OptionalIntArrayRef opt_dims, int64_t ndim) {
  if (opt_dims.has_value()) {
    return DimVector(opt_dims.value());
  } else {
    std::vector<int64_t> all_dims(ndim);
    std::iota(all_dims.begin(), all_dims.end(), 0);
    return DimVector(all_dims);
  }
}

inline DimMask make_dim_mask(OptionalIntArrayRef opt_dims, int64_t ndim, bool allow_empty_dims=false) {
  DimMask mask;
  if (opt_dims.has_value()) {
    auto dims = opt_dims.value();
    if (dims.empty() && !allow_empty_dims) {
      mask = DimMask().flip();
    } else {
      mask = at::dim_list_to_bitset(dims, ndim);
    }
  } else {
    mask = DimMask().flip();
  }
  return mask;
}

inline DimVector shape_from_dim_mask(const Tensor& self, DimMask mask, bool keepdim) {
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
  return shape;
}

inline void resize_reduction_result(
    Tensor& result, const Tensor& self, DimMask mask, bool keepdim,
    ScalarType /*dtype*/)
{
  auto shape = shape_from_dim_mask(self, mask, keepdim);
  TORCH_CHECK(result.defined(), "Cannot create a new tensor inside a reduction op. You likely tried to call an operator with an out argument but the out argument was an undefined tensor.");
  at::native::resize_output(result, shape);
}

inline Tensor create_reduction_result(
  const Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, ScalarType dtype
) {
  DimMask mask = make_dim_mask(dim, self.dim());
  auto shape = shape_from_dim_mask(self, mask, keepdim);
  return at::empty(shape, self.options().dtype(dtype));
}

inline Tensor review_reduce_result(const Tensor& result, int ndim, DimMask mask, bool keepdim) {
  if (keepdim) {
    return result;
  }
  auto shape = DimVector(result.sizes());
  auto stride = DimVector(result.strides());
  for (const auto dim : c10::irange(ndim)) {
    if (mask[dim]) {
      shape.insert(shape.begin() + dim, 1);
      stride.insert(stride.begin() + dim, 0);
    }
  }
  return result.as_strided(shape, stride);
}

inline TensorIterator make_reduction(
    const char* name, Tensor& result, const Tensor& self,
    at::OptionalIntArrayRef dim_opt,
    bool keepdim, ScalarType in_dtype, ScalarType out_dtype) {
  // check that result type and dtype match if provided
  TORCH_CHECK(
      !result.defined() || result.scalar_type() == out_dtype,
      name, ": provided dtype must match dtype of result. Got ",
      toString(result.scalar_type()),
      " and ",
      toString(out_dtype),
      ".");
  // dim={} performs an all-reduce, same as dim=None
  IntArrayRef dim = dim_opt.value_or(IntArrayRef{});
  int64_t ndim = self.dim();
  auto mask = make_dim_mask(dim, ndim);
  resize_reduction_result(result, self, mask, keepdim, out_dtype);
  auto viewed_result = review_reduce_result(result, ndim, mask, keepdim);
  namedinference::propagate_names_for_reduction(result, self, dim, keepdim);
  if (self.scalar_type() == in_dtype) {
    return TensorIterator::reduce_op(viewed_result, self);
  }
  return TensorIterator::reduce_op(viewed_result, self.to(in_dtype));
}

[[maybe_unused]] inline TensorIterator make_reduction(
    const char* name,
    Tensor& result,
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    ScalarType out_dtype) {
  // special case for type promotion in mixed precision, improves computational
  // efficiency.
  // not generalize this to common mismatched input/output types to avoid cross
  // product of templated kernel launches.
  const bool gpu_lowp_to_f32 = (
        (self.is_cuda() || self.is_xpu()) && (self.scalar_type() == kHalf || self.scalar_type() == kBFloat16) && out_dtype == kFloat);
  auto in_dtype = gpu_lowp_to_f32 ? self.scalar_type()
                   : self.is_complex() ? c10::toComplexType(out_dtype)
                                       : out_dtype;
  return make_reduction(name, result, self, dim, keepdim, in_dtype, out_dtype);
}

inline TensorIterator make_reduction(
    const char* name, Tensor& result1, Tensor& result2, const Tensor& self,
    at::OptionalIntArrayRef dim_opt, bool keepdim, ScalarType dtype1,
    ScalarType dtype2) {
  // check that result type and dtype match if provided
  TORCH_CHECK(
    (!result1.defined() || result1.scalar_type() == dtype1) && (!result2.defined() || result2.scalar_type() == dtype2),
    name, ": provided dtype must match dtype of result. Got ",
    toString(result1.scalar_type()), toString(result2.scalar_type()),
    " and ",
    toString(dtype1), toString(dtype2),
    ".");

  // dim={} performs an all-reduce, same as dim=None
  auto dim = dim_opt.value_or(IntArrayRef{});
  int64_t ndim = self.dim();
  DimMask mask = make_dim_mask(dim, ndim);
  resize_reduction_result(result1, self, mask, keepdim, dtype1);
  auto viewed_result1 = review_reduce_result(result1, ndim, mask, keepdim);

  resize_reduction_result(result2, self, mask, keepdim, dtype2);
  auto viewed_result2 = review_reduce_result(result2, ndim, mask, keepdim);

  namedinference::propagate_names_for_reduction(result1, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(result2, self, dim, keepdim);

  // special case for type promotion in mixed precision, improves computational
  // efficiency.
  // We don't generalize this to common mismatched input/output types to avoid cross
  // product of templated kernel launches.
  if (self.scalar_type() == dtype1 ||
      (self.is_cuda() && self.scalar_type() == kHalf && dtype1 == kFloat)) {
    return TensorIterator::reduce_op(viewed_result1, viewed_result2, self);
  }
  return TensorIterator::reduce_op(viewed_result1, viewed_result2, self.to(dtype1));
}

[[maybe_unused]] inline TensorIterator make_reduction(
    const char* name,
    Tensor& result1,
    Tensor& result2,
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    ScalarType dtype) {
  return make_reduction(name, result1, result2, self, dim, keepdim, dtype, dtype);
}

inline void zero_numel_check_dims(const Tensor& self, const int64_t dim, const char *fn_name) {
  if (self.ndimension() == 0) {
    TORCH_CHECK_INDEX(dim == 0 || dim == -1, fn_name,
      ": Expected reduction dim -1 or 0 for scalar but got ", dim);
  }
  else {
    TORCH_CHECK_INDEX(self.size(dim) != 0, fn_name,
      ": Expected reduction dim ", dim, " to have non-zero size.");
  }
}

inline void zero_numel_check_dims(const Tensor& self, const IntArrayRef dim, const char *fn_name) {
  TORCH_CHECK(
    !dim.empty(),
      fn_name, ": Expected reduction dim to be specified for input.numel() == 0. ",
        "Specify the reduction dim with the 'dim' argument.");
  for (const int64_t d : dim) {
    zero_numel_check_dims(self, d, fn_name);
  }
}

inline std::vector<int64_t> get_zero_numel_tensor_size(
    const Tensor& self,
    const int64_t dim,
    const bool keepdim,
    const char* fn_name) {
  TORCH_INTERNAL_ASSERT(self.numel() == 0,  fn_name, ": Expected self.numel() == 0.");
  zero_numel_check_dims(self, dim, fn_name);
  std::vector<int64_t> sizes;
  if (keepdim) {
    sizes = self.sizes().vec();
    sizes[dim] = 1;
  }
  else {
    for (const auto d : c10::irange(self.dim())) {
      if (d != dim) {
        sizes.push_back(self.sizes()[d]);
      }
    }
  }
  return sizes;
}

// Resize the result tensor and indices when result.numel() == 0 depending on values of
// dim and keepdim for returning tensors containing reduction results.
// This function should be called when you are reducing a zero-numel tensor and want to
// resize the output and return it. This function exists for resizing zero-numel
// tensors when the size of the reduction dimension is non-zero.
[[maybe_unused]] inline void zero_numel_tensor_resize(
    Tensor& result,
    Tensor& result_indices,
    const Tensor& self,
    const int64_t dim,
    const bool keepdim,
    const char* fn_name) {
  auto sizes = get_zero_numel_tensor_size(self, dim, keepdim, fn_name);
  at::native::resize_output(result, sizes);
  at::native::resize_output(result_indices, sizes);
}

inline ScalarType get_dtype_from_self(
    const Tensor& self,
    const std::optional<ScalarType>& dtype,
    bool promote_integers) {
  if (dtype.has_value()) {
    return dtype.value();
  }
  ScalarType src_type = self.scalar_type();
  if (promote_integers && at::isIntegralType(src_type, /*includeBool=*/true)) {
    return kLong;
  }
  return src_type;
}

inline ScalarType get_dtype_from_result(Tensor& result, std::optional<ScalarType> dtype) {
  TORCH_CHECK(result.defined(), "Cannot create a new tensor inside a reduction op. You likely tried to call an operator with an out argument but the out argument was an undefined tensor.");
  if (dtype.has_value()) {
    return dtype.value();
  } else {
    return result.scalar_type();
  }
}


} // namespace at::native

namespace at::meta {

[[maybe_unused]] inline DimVector get_reduction_shape(
    const Tensor& self,
    IntArrayRef dims,
    bool keepdim,
    bool allow_empty_dims = false) {
  auto mask = native::make_dim_mask(dims, self.dim(), allow_empty_dims);
  return native::shape_from_dim_mask(self, mask, keepdim);
}

inline void resize_reduction(
    impl::MetaBase& meta,
    const Tensor& self,
    OptionalIntArrayRef opt_dims,
    bool keepdim,
    ScalarType out_dtype,
    bool allow_empty_dims=false) {
  DimVector dims_ = at::native::make_dim_vector(opt_dims, self.dim());
  maybe_wrap_dims(dims_, self.dim());
  auto shape = get_reduction_shape(self, dims_, keepdim, allow_empty_dims);
  if (self.layout() == kStrided) {
    meta.set_output_raw_strided(0, shape, {}, self.options().dtype(out_dtype));
  } else if (shape.empty()) {
    meta.set_output_raw_strided(0, shape, {}, self.options().dtype(out_dtype).layout(kStrided));
  } else {
    TORCH_CHECK(false, "resize_reduction: support for output with ", self.layout(), " layout is not implemented yet");
  }
  namedinference::propagate_names_for_reduction(
      meta.maybe_get_output(), self, dims_, keepdim);
}

inline void resize_reduction_with_indices(
    impl::MetaBase& meta,
    const Tensor& self,
    IntArrayRef dims,
    bool keepdim,
    ScalarType out_dtype) {
  DimVector dims_(dims);
  maybe_wrap_dims(dims_, self.dim());
  auto shape = get_reduction_shape(self, dims_, keepdim);
  meta.set_output_raw_strided(0, shape, {}, self.options().dtype(out_dtype));
  meta.set_output_raw_strided(1, shape, {}, self.options().dtype(kLong));
  namedinference::propagate_names_for_reduction(
      meta.maybe_get_output(0), self, dims_, keepdim);
  namedinference::propagate_names_for_reduction(
      meta.maybe_get_output(1), self, dims_, keepdim);
}

inline TensorIterator make_reduction(
    const Tensor& self,
    const Tensor& result,
    OptionalIntArrayRef opt_dims,
    bool keepdim,
    ScalarType in_dtype) {
  int64_t ndim = self.dim();
  auto mask = at::native::make_dim_mask(opt_dims, ndim);
  auto viewed_result =
      at::native::review_reduce_result(result, ndim, mask, keepdim);
  if (self.scalar_type() == in_dtype) {
    return TensorIterator::reduce_op(viewed_result, self);
  }
  return TensorIterator::reduce_op(viewed_result, self.to(in_dtype));
}

inline TensorIterator make_reduction(
    const Tensor& self,
    const Tensor& result1,
    const Tensor& result2,
    IntArrayRef dims,
    bool keepdim,
    ScalarType dtype1,
    ScalarType /*dtype2*/) {
  int64_t ndim = self.dim();
  auto mask = at::native::make_dim_mask(dims, ndim);
  auto viewed_result1 = at::native::review_reduce_result(result1, ndim, mask, keepdim);
  auto viewed_result2 = at::native::review_reduce_result(result2, ndim, mask, keepdim);
  // special case for type promotion in mixed precision, improves computational efficiency.
  // We don't generalize this to common mismatched input/output types to avoid cross product
  // of templated kernel launches.
  if (self.scalar_type() == dtype1 ||
      (self.is_cuda() && self.scalar_type() == kHalf && dtype1 == kFloat)) {
    return TensorIterator::reduce_op(viewed_result1, viewed_result2, self);
  }
  return TensorIterator::reduce_op(viewed_result1, viewed_result2, self.to(dtype1));
}

[[maybe_unused]] inline TensorIterator make_reduction_from_out_ty(
    const Tensor& self,
    const Tensor& result,
    OptionalIntArrayRef opt_dims,
    bool keepdim,
    ScalarType out_dtype) {
  // special case for type promotion in mixed precision, improves computational
  // efficiency.
  // not generalize this to common mismatched input/output types to avoid cross
  // product of templated kernel launches.
  const bool gpu_lowp_to_f32 =
      (self.is_cuda() &&
       (self.scalar_type() == kHalf || self.scalar_type() == kBFloat16) &&
       out_dtype == kFloat);
  auto in_dtype = gpu_lowp_to_f32 ? self.scalar_type() : out_dtype;
  return make_reduction(self, result, opt_dims, keepdim, in_dtype);
}

} // namespace at::meta
