#pragma once

#include <limits>
#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/WrapDimUtilsMulti.h>

namespace at { namespace native {

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

static inline int64_t ensure_nonempty_dim(int64_t dim) {
  return std::max<int64_t>(dim, 1);
}

static inline int64_t ensure_nonempty_size(const Tensor& t, int64_t dim) {
  return t.dim() == 0 ? 1 : t.size(dim);
}

static inline int64_t ensure_nonempty_stride(const Tensor& t, int64_t dim) {
  return t.dim() == 0 ? 1 : t.stride(dim);
}

using IdxVec = std::vector<int64_t>;
static inline IdxVec ensure_nonempty_vec(IdxVec vec) {
  if (vec.size() == 0) {
    vec.push_back(1);
  }
  return vec;
}

static inline Tensor restride_dim(
  const Tensor& src, int64_t dim,
  IntArrayRef replacement_shape
) {
  auto strides = ensure_nonempty_vec(src.strides().vec());
  strides[dim] = 0;
  return src.as_strided(replacement_shape, strides);
}

inline Tensor &_dimreduce_setup(Tensor &result, const Tensor &self,
                                int64_t dim) {
  IntArrayRef self_sizes = self.sizes();
  std::vector<int64_t> result_sizes;
  result_sizes.insert(result_sizes.end(), self_sizes.begin(), self_sizes.end());
  result_sizes[dim] = 1;
  result.resize_(result_sizes);
  return result;
}

inline bool _dimreduce_return_trivial(Tensor &result, const Tensor &self,
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
                                               int64_t dim, bool keepdim, const char *fn_name) {
  if (self.numel() == 1 && self.ndimension() == 0) {
    result.resize_({});
    result.fill_(self);
    return true;
  }

  if (self.numel() == 0) {
    AT_ERROR("cannot perform reduction function ", fn_name,
             " on tensor with no elements because the operation does not have an identity");
  }
  return false;
}

inline c10::optional<Tensor> _allreduce_return_trivial(
    const Tensor& self,
    const Scalar& ident) {
  // Return identity
  if (self.numel() == 0) {
    return at::scalar_tensor(ident, self.options());
  }
  return c10::nullopt;
}

#define OPTION_TYPE_EQUALITY_CHECK(option, out, self) \
{ \
  TORCH_CHECK(\
    out.option() == self.option(),\
    "expected ", #option, " ",\
    self.option(),\
    " but found ", out.option())\
}

static inline void check_scalar_type_device_layout_equal(const Tensor& out, const Tensor& self) {
  OPTION_TYPE_EQUALITY_CHECK(scalar_type, out, self);
  OPTION_TYPE_EQUALITY_CHECK(device, out.options(), self.options());
  OPTION_TYPE_EQUALITY_CHECK(layout, out.options(), self.options());
}

static inline Tensor integer_upcast(const Tensor& self, optional<ScalarType> dtype) {
  ScalarType scalarType = self.scalar_type();
  ScalarType upcast_scalarType = dtype.value_or(at::isIntegralType(scalarType, /*includeBool=*/true) ? ScalarType::Long : scalarType);
  return self.toType(upcast_scalarType);
}

using DimMask = TensorIterator::DimMask;

static DimMask make_dim_mask(IntArrayRef dims, int64_t ndim) {
  DimMask mask;
  if (dims.empty()) {
    mask = DimMask().flip();
  } else {
    mask = at::dim_list_to_bitset(dims, ndim);
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
    result = at::empty(shape, self.options().dtype(dtype));
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

static TensorIterator make_reduction(
    const char* name, Tensor& result, const Tensor& self, IntArrayRef dim,
    bool keepdim, ScalarType in_dtype, ScalarType out_dtype)
{
  // check that result type and dtype match if provided
  TORCH_CHECK(
      !result.defined() || result.scalar_type() == out_dtype,
      name, ": provided dtype must match dtype of result. Got ",
      toString(result.scalar_type()),
      " and ",
      toString(out_dtype),
      ".");
  int64_t ndim = self.dim();
  auto mask = make_dim_mask(dim, ndim);
  allocate_reduction_result(result, self, mask, keepdim, out_dtype);
  auto viewed_result = review_reduce_result(result, ndim, mask, keepdim);
  namedinference::propagate_names_for_reduction(result, self, dim, keepdim);
  if (self.scalar_type() == in_dtype) {
    return TensorIterator::reduce_op(viewed_result, self);
  }
  return TensorIterator::reduce_op(viewed_result, self.to(in_dtype));
}

static TensorIterator make_reduction(
    const char* name, Tensor& result, const Tensor& self, IntArrayRef dim,
    bool keepdim, ScalarType out_dtype)
{
  // special case for type promotion in mixed precision, improves computational
  // efficiency.
  // not generalize this to common mismatched input/output types to avoid cross
  // product of templated kernel launches.
  const bool gpu_f16_to_f32 = (
    self.is_cuda() && self.scalar_type() == kHalf && out_dtype == kFloat);
  auto in_dtype = gpu_f16_to_f32 ? self.scalar_type() : out_dtype;
  return make_reduction(name, result, self, dim, keepdim, in_dtype, out_dtype);
}

static TensorIterator make_reduction(
    const char* name, Tensor& result1, Tensor& result2, const Tensor& self, IntArrayRef dim,
    bool keepdim, ScalarType dtype1, ScalarType dtype2)
{
  // check that result type and dtype match if provided
  TORCH_CHECK(
    (!result1.defined() || result1.scalar_type() == dtype1) && (!result2.defined() || result2.scalar_type() == dtype2),
    name, ": provided dtype must match dtype of result. Got ",
    toString(result1.scalar_type()), toString(result2.scalar_type()),
    " and ",
    toString(dtype1), toString(dtype2),
    ".");

  int64_t ndim = self.dim();
  DimMask mask = make_dim_mask(dim, ndim);
  allocate_reduction_result(result1, self, mask, keepdim, dtype1);
  auto viewed_result1 = review_reduce_result(result1, ndim, mask, keepdim);

  allocate_reduction_result(result2, self, mask, keepdim, dtype2);
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

static TensorIterator make_reduction(
    const char* name, Tensor& result1, Tensor& result2, const Tensor& self, IntArrayRef dim,
    bool keepdim, ScalarType dtype)
{
  return make_reduction(name, result1, result2, self, dim, keepdim, dtype, dtype);
}

}}  // at::native
