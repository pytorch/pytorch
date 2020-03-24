#include <ATen/native/ReduceOps.h>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/TensorDimApply.h>
#include <ATen/native/SharedReduceOps.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>
#include <map>
#include <cmath>
#include <cfloat>
#include <type_traits>

namespace at {
namespace native {

DEFINE_DISPATCH(sum_stub);
DEFINE_DISPATCH(std_var_stub);
DEFINE_DISPATCH(prod_stub);
DEFINE_DISPATCH(norm_stub);
DEFINE_DISPATCH(mean_stub);
DEFINE_DISPATCH(and_stub);
DEFINE_DISPATCH(or_stub);
DEFINE_DISPATCH(min_values_stub);
DEFINE_DISPATCH(max_values_stub);
DEFINE_DISPATCH(argmax_stub);
DEFINE_DISPATCH(argmin_stub);
DEFINE_DISPATCH(cumsum_stub);
DEFINE_DISPATCH(cumprod_stub);

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
  auto mask = DimMask();
  if (dims.empty()) {
    mask.flip();
  } else {
    for (int64_t dim : dims) {
      int64_t pos_dim = maybe_wrap_dim(dim, ndim);
      TORCH_CHECK(pos_dim < 64, "PyTorch doesn't support reduction operations for dim>=64");
      mask.set(pos_dim);
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
    bool keepdim, ScalarType dtype)
{
  // check that result type and dtype match if provided
  for (const Tensor *t: {&result1, &result2}) {
    const Tensor& result = *t;
    TORCH_CHECK(
        !result.defined() || result.scalar_type() == dtype,
        name, ": provided dtype must match dtype of result. Got ",
        toString(result.scalar_type()),
        " and ",
        toString(dtype),
        ".");
  }

  int64_t ndim = self.dim();
  DimMask mask = make_dim_mask(dim, ndim);
  allocate_reduction_result(result1, self, mask, keepdim, dtype);
  auto viewed_result1 = review_reduce_result(result1, ndim, mask, keepdim);

  allocate_reduction_result(result2, self, mask, keepdim, dtype);
  auto viewed_result2 = review_reduce_result(result2, ndim, mask, keepdim);

  namedinference::propagate_names_for_reduction(result1, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(result2, self, dim, keepdim);

  // special case for type promotion in mixed precision, improves computational
  // efficiency.
  // We don't generalize this to common mismatched input/output types to avoid cross
  // product of templated kernel launches.
  if (self.scalar_type() == dtype ||
      (self.is_cuda() && self.scalar_type() == kHalf && dtype == kFloat)) {
    return TensorIterator::reduce_op(viewed_result1, viewed_result2, self);
  }
  return TensorIterator::reduce_op(viewed_result1, viewed_result2, self.to(dtype));
}

Tensor _cumsum_cpu(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  cumsum_stub(self.device().type(), result, self, dim);
  return result;
}

Tensor& _cumsum_out_cpu(Tensor& result, const Tensor& self, int64_t dim) {
  cumsum_stub(self.device().type(), result, self, dim);
  return result;
}

Tensor cumsum(const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
  auto result = [&]() {
    NoNamesGuard guard;
    return at::_cumsum(integer_upcast(self, dtype), dim);
  }();
  namedinference::propagate_names(result, self);
  return result;
}

Tensor& cumsum_out(Tensor& result, const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
  // result type is favored over dtype; check that they match if provided (NumPy doesn't check)
  TORCH_CHECK(
      !dtype.has_value() || (result.scalar_type() == dtype.value()),
      "provided dtype must match dtype of result in cumsum. Got ",
      toString(result.scalar_type()),
      " and ",
      toString(dtype.value()),
      ".");
  {
    NoNamesGuard guard;
    at::_cumsum_out(result, self.toType(result.scalar_type()), dim);
  }
  namedinference::propagate_names(result, self);
  return result;
}

Tensor _cumprod_cpu(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  cumprod_stub(self.device().type(), result, self, dim);
  return result;
}

Tensor& _cumprod_out_cpu(Tensor& result, const Tensor& self, int64_t dim) {
  cumprod_stub(self.device().type(), result, self, dim);
  return result;
}

Tensor cumprod(const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
  auto result = [&]() {
    NoNamesGuard guard;
    return at::_cumprod(integer_upcast(self, dtype), dim);
  }();
  namedinference::propagate_names(result, self);
  return result;
}

Tensor& cumprod_out(Tensor& result, const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
  // result type is favored over dtype; check that they match if provided (NumPy doesn't check)
  TORCH_CHECK(
      !dtype.has_value() || (result.scalar_type() == dtype.value()),
      "provided dtype must match dtype of result in cumprod. Got ",
      toString(result.scalar_type()),
      " and ",
      toString(dtype.value()),
      ".");
  {
    NoNamesGuard guard;
    at::_cumprod_out(result, self.toType(result.scalar_type()), dim);
  }
  namedinference::propagate_names(result, self);
  return result;
}

// Implement std::is_nan<IntegralType> for MSVC.
namespace {
#ifdef _MSC_VER
template<typename T>
inline typename std::enable_if<std::is_integral<T>::value, bool>::type isnan_(T x) {
  return false;
}
template<typename T>
inline typename std::enable_if<!std::is_integral<T>::value, bool>::type isnan_(T x) {
  return std::isnan(x);
}
#else
template<typename T>
inline bool isnan_(T x) {
  return std::isnan(x);
}
#endif
}

template<typename T1, typename T2, typename Operation>
void cummax_cummin_helper(const T1* self_data, T1* values_data, T2* indices_data,
          int self_dim_size, int self_stride, int values_stride, int indices_stride) {
      Operation op;
      T1 out = self_data[0];
      int idx = 0;
      for(int i = 0; i < self_dim_size; i++) {
        T1 curr_elem = self_data[i*self_stride];
        if(isnan_(curr_elem) || (!isnan_(out) && op(curr_elem, out))) {
            out = self_data[i*self_stride];
            idx = i;
        }
        values_data[i*values_stride] = out;
        indices_data[i*indices_stride] = idx;
      }
}

void cummax_helper_cpu(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool,
    self.scalar_type(), "cummax_cpu",
    [&] {
      at::native::tensor_dim_apply3<scalar_t, int64_t>(self, values, indices, dim, cummax_cummin_helper<scalar_t, int64_t, std::greater_equal<scalar_t>>);
    });
}

std::tuple<Tensor&, Tensor&> cummax_out(Tensor& values, Tensor& indices, const Tensor& self, int64_t dim) {
  check_scalar_type_device_layout_equal(values, self);
  check_scalar_type_device_layout_equal(indices, at::empty({0}, self.options().dtype(at::kLong)));
  {
    NoNamesGuard guard;
    values.resize_(self.sizes());
    indices.resize_(self.sizes());
    if(self.dim() == 0) {
      values.fill_(self);
      indices.fill_(0);
    } else if(self.numel() != 0) {
      dim = maybe_wrap_dim(dim, self.dim());
      at::_cummax_helper(self, values, indices, dim);
    }
  }
  namedinference::propagate_names(values, self);
  namedinference::propagate_names(indices, self);
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> cummax(const Tensor& self, int64_t dim) {
  auto values = at::empty(self.sizes(), self.options());
  auto indices = at::empty(self.sizes(), self.options().dtype(at::kLong));
  at::cummax_out(values, indices, self, dim);
  return std::make_tuple(values, indices);
}

void cummin_helper_cpu(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool,
    self.scalar_type(), "cummin_cpu",
    [&] {
      at::native::tensor_dim_apply3<scalar_t, int64_t>(self, values, indices, dim, cummax_cummin_helper<scalar_t, int64_t, std::less_equal<scalar_t>>);
    });
}

std::tuple<Tensor&, Tensor&> cummin_out(Tensor& values, Tensor& indices, const Tensor& self, int64_t dim) {
  check_scalar_type_device_layout_equal(values, self);
  check_scalar_type_device_layout_equal(indices, at::empty({0}, self.options().dtype(at::kLong)));
  {
    NoNamesGuard guard;
    values.resize_(self.sizes());
    indices.resize_(self.sizes());
    if(self.dim() == 0) {
      values.fill_(self);
      indices.fill_(0);
    } else if(self.numel() != 0) {
      dim = maybe_wrap_dim(dim, self.dim());
      at::_cummin_helper(self, values, indices, dim);
    }
  }
  namedinference::propagate_names(values, self);
  namedinference::propagate_names(indices, self);
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> cummin(const Tensor& self, int64_t dim) {
  auto values = at::empty(self.sizes(), self.options());
  auto indices = at::empty(self.sizes(), self.options().dtype(at::kLong));
  at::cummin_out(values, indices, self, dim);
  return std::make_tuple(values, indices);
}
// ALL REDUCE #################################################################

static ScalarType get_dtype(Tensor& result, const Tensor& self, optional<ScalarType> dtype,
                            bool promote_integers=false) {
  if (dtype.has_value()) {
    return dtype.value();
  } else if (result.defined()) {
    return result.scalar_type();
  }
  ScalarType src_type = self.scalar_type();
  if (promote_integers && at::isIntegralType(src_type, /*includeBool=*/true)) {
    return kLong;
  }
  return src_type;
}

Tensor& sum_out(Tensor& result, const Tensor& self, IntArrayRef dim,
                       bool keepdim, optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  auto iter = make_reduction("sum", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    sum_stub(iter.device_type(), iter);
  }
  return result;
}

Tensor sum(const Tensor &self, c10::optional<ScalarType> dtype) {
  return at::native::sum(self, std::vector<int64_t>{}, false, dtype);
}
Tensor sum(const Tensor& self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  Tensor result;
  return at::native::sum_out(result, self, dim, keepdim, dtype);
}
Tensor sum(const Tensor& self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) {
  return at::sum(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor& sum_out(Tensor& result, const Tensor& self, DimnameList dim,
                bool keepdim, optional<ScalarType> opt_dtype) {
  return at::sum_out(result, self, dimnames_to_positions(self, dim), keepdim, opt_dtype);
}

static Tensor& prod_out_impl(Tensor& result, const Tensor& self, IntArrayRef dim,
                        bool keepdim, c10::optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  auto iter = make_reduction("prod", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.fill_(1);
  } else {
    prod_stub(iter.device_type(), iter);
  }
  return result;
}

Tensor prod(const Tensor& self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) {
  Tensor result;
  native::prod_out_impl(result, self, dim, keepdim, dtype);
  return result;
}

Tensor prod(const Tensor &self, c10::optional<ScalarType> dtype) {
  Tensor result;
  return at::native::prod_out_impl(result, self, {}, false, dtype);
}

Tensor& prod_out(Tensor& result, const Tensor& self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) {
  return at::native::prod_out_impl(result, self, dim, keepdim, dtype);
}

Tensor prod(const Tensor& self, Dimname dim, bool keepdim, c10::optional<ScalarType> dtype) {
  return at::prod(self, dimname_to_position(self, dim), keepdim, dtype);
}

Tensor& prod_out(Tensor& result, const Tensor& self, Dimname dim,
                 bool keepdim, optional<ScalarType> opt_dtype) {
  return at::prod_out(result, self, dimname_to_position(self, dim), keepdim, opt_dtype);
}

Tensor &mean_out_cpu_gpu(Tensor &result, const Tensor &self, IntArrayRef dim,
                 bool keepdim, c10::optional<ScalarType> opt_dtype) {
  ScalarType scalarType = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
  TORCH_CHECK(
      at::isFloatingType(scalarType) || at::isComplexType(scalarType),
      "Can only calculate the mean of floating types. Got ",
      toString(scalarType),
      " instead.");
  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  // TODO: the TensorIterator reduction implementation of mean
  // (mean_kernel_impl()) is unvectorized and leads to very poor performance
  // for production workloads. Once that's fixed, the following code can be used
  // in lieu of the sum + divide implementation below.
  if (self.device().is_cpu()) {
    int64_t dim_prod = 1;
    if (dim.size() == 0 || self.ndimension() == 0) {
      dim_prod = self.numel();
    } else {
      for (auto d : dim) {
        dim_prod *= self.size(d);
      }
    }
    at::sum_out(result, self, dim, keepdim, dtype).div_(dim_prod);
    return result;
  }

  auto iter = make_reduction("mean", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.fill_(std::numeric_limits<double>::quiet_NaN());
  } else {
    mean_stub(iter.device_type(), iter);
  }
  return result;
}

Tensor mean_cpu_gpu(const Tensor &self, optional<ScalarType> dtype) {
  return at::native::mean_cpu_gpu(self, IntArrayRef{}, false, dtype);
}

Tensor mean_cpu_gpu(const Tensor& self, IntArrayRef dim, bool keepdim, optional<ScalarType> dtype) {
  Tensor result;
  return at::native::mean_out_cpu_gpu(result, self, dim, keepdim, dtype);
}

Tensor mean(const Tensor& self, DimnameList dim, bool keepdim, optional<ScalarType> dtype) {
  return at::mean(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor& mean_out(Tensor& result, const Tensor& self, DimnameList dim,
                 bool keepdim, c10::optional<ScalarType> opt_dtype) {
  return at::mean_out(result, self, dimnames_to_positions(self, dim), keepdim, opt_dtype);
}

static Tensor squeeze_multiple(const Tensor& self, IntArrayRef dims) {
  int ndims = self.sizes().size();
  auto dims_to_squeeze = at::dim_list_to_bitset(dims, ndims);
  Tensor result = self;
  for (int i = ndims - 1; i >= 0; --i) {
    if (dims_to_squeeze[i]) {
      result = result.squeeze(i);
    }
  }
  return result;
}

static Tensor& logsumexp_out_impl(Tensor& result, const Tensor& self, IntArrayRef dims, bool keepdim) {
  // can't take max of empty tensor
  if (self.numel() != 0) {
    auto maxes = at::max_values(self, dims, true);
    auto maxes_squeezed = (keepdim ? maxes : squeeze_multiple(maxes, dims));
    maxes_squeezed.masked_fill_(maxes_squeezed.abs() == INFINITY, 0);
    at::sum_out(result, at::exp(self - maxes), dims, keepdim);
    result.log_().add_(maxes_squeezed);
  } else {
    at::sum_out(result, at::exp(self), dims, keepdim);
    result.log_();
  }
  return result;
}

Tensor& logsumexp_out(Tensor& result, const Tensor& self, IntArrayRef dims, bool keepdim) {
  {
    NoNamesGuard guard;
    logsumexp_out_impl(result, self, dims, keepdim);
  }
  namedinference::propagate_names_for_reduction(result, self, dims, keepdim);
  return result;
}

Tensor logsumexp(const Tensor& self, IntArrayRef dims, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::logsumexp_out(result, self, dims, keepdim);
}

Tensor logsumexp(const Tensor& self, DimnameList dims, bool keepdim) {
  return at::logsumexp(self, dimnames_to_positions(self, dims), keepdim);
}

Tensor& logsumexp_out(Tensor& result, const Tensor& self, DimnameList dims, bool keepdim) {
  return at::logsumexp_out(result, self, dimnames_to_positions(self, dims), keepdim);
}

static Tensor& norm_out(Tensor &result, const Tensor &self, optional<Scalar> opt_p,
                               IntArrayRef dim, bool keepdim, optional<ScalarType> opt_dtype) {
  auto p = opt_p.value_or(2.0);
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "norm only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "norm only supports strided layout, got: ", self.layout());

  ScalarType scalarType = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
  TORCH_CHECK(
      at::isFloatingType(scalarType) || at::isComplexType(scalarType),
      "Can only calculate the mean of floating types. Got ",
      toString(scalarType),
      " instead.");

  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  auto iter = make_reduction("norm", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    norm_stub(iter.device_type(), iter, p);
  }
  return result;
}

static inline Tensor _norm(const Tensor &self, Scalar p) {
  if (self.is_sparse()) {
    return at::native_norm(self, p);
  } else {
    TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
                "norm only supports CPU AND CUDA device type, got: ", self.device().type());
    TORCH_CHECK(self.layout() == Layout::Strided,
                "norm only supports strided layout, got: ", self.layout());
    TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
                "norm only supports floating-point dtypes");

    Tensor result;
    return at::native::norm_out(result, self, p, IntArrayRef{}, false, c10::nullopt);
  }
}

Tensor &norm_out(Tensor& result, const Tensor& self, optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) {
  return at::native::norm_out(result, self, p, dim, keepdim, optional<ScalarType>(dtype));
}

Tensor &norm_out(Tensor& result, const Tensor& self, optional<Scalar> p, IntArrayRef dim, bool keepdim) {
  return at::native::norm_out(result, self, p, dim, keepdim, c10::nullopt);
}

static Tensor norm(const Tensor& self, optional<Scalar> p, IntArrayRef dim, bool keepdim,
            optional<ScalarType> opt_dtype) {
  Tensor result;
  return at::native::norm_out(result, self, p, dim, keepdim, opt_dtype);
}

Tensor norm(const Tensor& self, optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) {
  return at::native::norm(self, p, dim, keepdim, optional<ScalarType>(dtype));
}

Tensor norm(const Tensor& self, optional<Scalar> p, ScalarType dtype) {
  return at::native::norm(self, p, IntArrayRef{}, false, optional<ScalarType>(dtype));
}

Tensor norm(const Tensor& self, optional<Scalar> p, IntArrayRef dim, bool keepdim) {
  return at::native::norm(self, p, dim, keepdim, c10::nullopt);
}

// leave it so we support sparse tensors
Tensor norm(const Tensor& self, Scalar p) {
  return at::native::_norm(self, p);
}

inline Tensor & _all(Tensor & result, TensorIterator & iter) {
  if (iter.numel() == 0) {
    result.fill_(1);
  } else {
    and_stub(iter.device_type(), iter);
  }

  return result;
}

Tensor all(const Tensor& self) {
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "all only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "all only supports strided layout, got: ", self.layout());
  TORCH_CHECK(self.scalar_type() == at::ScalarType::Byte || self.scalar_type() == at::ScalarType::Bool,
    "all only supports torch.uint8 and torch.bool dtypes");

  Tensor result = at::empty({0}, self.options());
  auto iter = make_reduction(
    "all", result, self, {}, false, self.scalar_type());
  return _all(result, iter);
}

Tensor all(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::all_out(result, self, dim, keepdim);
}

Tensor &all_out(Tensor &result, const Tensor &self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "all only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "all only supports strided layout, got: ", self.layout());
  TORCH_CHECK(self.scalar_type() == at::ScalarType::Byte || self.scalar_type() == at::ScalarType::Bool,
    "all only supports torch.uint8 and torch.bool dtypes");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial(result, self, 1, dim, keepdim)) {
    return result;
  } else {
    auto iter = make_reduction(
      "all", result, self, dim, keepdim, self.scalar_type());
    return _all(result, iter);
  }
}

inline Tensor & _any(Tensor & result, TensorIterator & iter) {
  if (iter.numel() == 0) {
    result.fill_(0);
  } else {
    or_stub(iter.device_type(), iter);
  }

  return result;
}

Tensor any(const Tensor& self) {
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "any only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided || self.layout() == Layout::Sparse,
              "any only supports strided AND sparse layout, got: ", self.layout());
  TORCH_CHECK(self.scalar_type() == at::ScalarType::Byte || self.scalar_type() == at::ScalarType::Bool,
    "all only supports torch.uint8 and torch.bool dtypes");

  Tensor result = at::empty({0}, self.options());
  auto iter = make_reduction(
    "any", result, self, {}, false, self.scalar_type());
  return _any(result, iter);
}

Tensor any(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::any_out(result, self, dim, keepdim);
}

Tensor &any_out(Tensor &result, const Tensor &self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "any only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "any only supports strided layout, got: ", self.layout());
  TORCH_CHECK(self.scalar_type() == at::ScalarType::Byte || self.scalar_type() == at::ScalarType::Bool,
    "all only supports torch.uint8 and torch.bool dtypes");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial(result, self, 0, dim, keepdim)) {
    return result;
  } else {
    auto iter = make_reduction(
      "any", result, self, dim, keepdim, self.scalar_type());
    return _any(result, iter);
  }
}

Tensor min_values(const Tensor& self, IntArrayRef dims, bool keepdim) {
  if (dims.size() == 1) {
    return std::get<0>(self.min(dims[0], keepdim));
  } else {
    Tensor result = at::empty({0}, self.options());
    ScalarType dtype = get_dtype(result, self, {}, true);
    auto iter = make_reduction("min_values", result, self, dims, keepdim, dtype);
    TORCH_CHECK(iter.numel() > 0, "min_values on a tensor with no elements is not defined.");
    min_values_stub(iter.device_type(), iter);
    return result;
  }
}

Tensor max_values(const Tensor& self, IntArrayRef dims, bool keepdim) {
  if (dims.size() == 1) {
    return std::get<0>(self.max(dims[0], keepdim));
  } else {
    Tensor result = at::empty({0}, self.options());
    ScalarType dtype = get_dtype(result, self, {}, true);
    auto iter = make_reduction("max_values", result, self, dims, keepdim, dtype);
    TORCH_CHECK(iter.numel() > 0, "max_values on a tensor with no elements is not defined.");
    max_values_stub(iter.device_type(), iter);
    return result;
  }
}

Tensor min_values(const Tensor& self, DimnameList dims, bool keepdim) {
  TORCH_CHECK(false, "NYI: min_values with names");
  return at::min_values(self, dimnames_to_positions(self, dims), keepdim);
}

Tensor max_values(const Tensor& self, DimnameList dims, bool keepdim) {
  TORCH_CHECK(false, "NYI: max_values with names");
  return at::max_values(self, dimnames_to_positions(self, dims), keepdim);
}

Tensor& argmax_out(Tensor& result, const Tensor& self, c10::optional<int64_t> dim, bool keepdim) {
  TORCH_CHECK(self.numel() > 0, "cannot perform reduction function argmax on a "
      "tensor with no elements because the operation does not have an identity");
  Tensor in;
  if (dim) {
    in = self;
  } else {
    in = self.reshape({-1});
    keepdim = false;
  }
  auto itr = make_reduction("argmax", result, in, dim.value_or(0), keepdim,
      self.scalar_type(), at::kLong);
  argmax_stub(itr.device_type(), itr);
  return result;
}

Tensor argmax(const Tensor& self, c10::optional<int64_t> dim, bool keepdims) {
  Tensor result = at::empty({0}, self.options().dtype(at::kLong));
  return at::native::argmax_out(result, self, dim, keepdims);
}

Tensor& argmin_out(Tensor& result, const Tensor& self, c10::optional<int64_t> dim, bool keepdim) {
  TORCH_CHECK(self.numel() > 0, "cannot perform reduction function argmin on a "
      "tensor with no elements because the operation does not have an identity");
  Tensor in;
  if (dim) {
    in = self;
  } else {
    in = self.reshape({-1});
    keepdim = false;
  }
  auto itr = make_reduction("argmin", result, in, dim.value_or(0), keepdim,
      self.scalar_type(), at::kLong);
  argmin_stub(itr.device_type(), itr);
  return result;
}

Tensor argmin(const Tensor& self, c10::optional<int64_t> dim, bool keepdims) {
  Tensor result = at::empty({0}, self.options().dtype(at::kLong));
  return at::native::argmin_out(result, self, dim, keepdims);
}

static Tensor &std_var_out(Tensor &result, const Tensor &self, IntArrayRef dim, bool unbiased, bool keepdim, bool take_sqrt) {
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "std and var only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "std and var only supports strided layout, got: ", self.layout());
  TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
              "std and var only support floating-point dtypes");

  if (at::isComplexType(self.scalar_type())){
    ScalarType dtype = c10::toValueType(get_dtype(result, self, {}, true));
    Tensor real_in = self.real().to(dtype);
    Tensor real_out = at::empty({0}, self.options().dtype(dtype));
    auto iter = make_reduction("std or var", real_out, real_in, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      real_out.fill_(NAN);
    } else {
      std_var_stub(iter.device_type(), iter, unbiased, false);
    }
    Tensor imag_in = self.imag().to(dtype);
    Tensor imag_out = at::empty({0}, self.options().dtype(dtype));
    iter = make_reduction("std or var", imag_out, imag_in, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      imag_out.fill_(NAN);
    } else {
      std_var_stub(iter.device_type(), iter, unbiased, false);
    }
    at::add_out(result, real_out, imag_out);
    take_sqrt ? at::sqrt_out(result, result) : result;
  } else{
    ScalarType dtype = get_dtype(result, self, {}, true);
    auto iter = make_reduction("std or var", result, self, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      result.fill_(NAN);
    } else {
      std_var_stub(iter.device_type(), iter, unbiased, take_sqrt);
    }
  }
  return result;
}

static std::tuple<Tensor&,Tensor&> std_var_mean_out(const char* fname, Tensor &result1, Tensor &result2, const Tensor &self, IntArrayRef dim, bool unbiased, bool keepdim, bool take_sqrt) {
  AT_ASSERT(result1.defined() && result2.defined());
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              fname, " only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              fname, " only supports strided layout, got: ", self.layout());
  TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
              fname, " only support floating-point dtypes");
  TORCH_CHECK(result1.scalar_type() == result2.scalar_type(),
           "provided by result1 dtype must match dtype of result2. Got ",
           toString(result1.scalar_type()),
           " and ",
           toString(result2.scalar_type()),
           ".");
  if (at::isComplexType(self.scalar_type())){
    ScalarType dtype = c10::toValueType(get_dtype(result1, self, {}, true));
    Tensor real_in = self.real().to(dtype);
    Tensor real_out_var = at::empty({0}, self.options().dtype(dtype));
    Tensor real_out_mean = at::empty({0}, self.options().dtype(dtype));
    auto iter = make_reduction(fname, real_out_var, real_out_mean, real_in, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      real_out_var.fill_(NAN);
      real_out_mean.fill_(NAN);
    } else {
      std_var_stub(iter.device_type(), iter, unbiased, false);
    }
    Tensor imag_in = self.imag().to(dtype);
    Tensor imag_out_var = at::empty({0}, self.options().dtype(dtype));
    Tensor imag_out_mean = at::empty({0}, self.options().dtype(dtype));
    iter = make_reduction(fname, imag_out_var, imag_out_mean, imag_in, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      imag_out_var.fill_(NAN);
      imag_out_mean.fill_(NAN);
    } else {
      std_var_stub(iter.device_type(), iter, unbiased, false);
    }
    at::add_out(result1, real_out_var, imag_out_var);
    take_sqrt ? at::sqrt_out(result1, result1) : result1;
    at::add_out(result2, real_out_mean, at::mul(imag_out_mean, std::complex<double>{0.0, 1.0}));
  } else {
    ScalarType dtype = get_dtype(result1, self, {}, true);
    auto iter = make_reduction(fname, result1, result2, self, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      result1.fill_(NAN);
      result2.fill_(NAN);
    } else {
      std_var_stub(iter.device_type(), iter, unbiased, take_sqrt);
    }
  }
  return std::tuple<Tensor&, Tensor&>(result1, result2);
}

std::tuple<Tensor&,Tensor&> var_mean_out(Tensor &result1, Tensor &result2, const Tensor &self, IntArrayRef dim, bool unbiased, bool keepdim) {
  return std_var_mean_out("var_mean", result1, result2, self, dim, unbiased, keepdim, false);
}

std::tuple<Tensor&,Tensor&> std_mean_out(Tensor &result1, Tensor &result2, const Tensor &self, IntArrayRef dim, bool unbiased, bool keepdim) {
  return std_var_mean_out("std_mean", result1, result2, self, dim, unbiased, keepdim, true);
}

std::tuple<Tensor&,Tensor&> var_mean_out(Tensor &result1, Tensor &result2, const Tensor &self, bool unbiased) {
  return std_var_mean_out("var_mean", result1, result2, self, {}, unbiased, false, false);
}

std::tuple<Tensor&,Tensor&> std_mean_out(Tensor &result1, Tensor &result2, const Tensor &self, bool unbiased) {
  return std_var_mean_out("std_mean", result1, result2, self, {}, unbiased, false, true);
}

std::tuple<Tensor,Tensor> var_mean(const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  Tensor result1 = at::empty({0}, self.options());
  Tensor result2 = at::empty({0}, self.options());
  return at::native::var_mean_out(result1, result2, self, dim, unbiased, keepdim);
}

std::tuple<Tensor,Tensor> std_mean(const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  Tensor result1 = at::empty({0}, self.options());
  Tensor result2 = at::empty({0}, self.options());
  return at::native::std_mean_out(result1, result2, self, dim, unbiased, keepdim);
}

std::tuple<Tensor,Tensor> std_mean(const Tensor& self, bool unbiased) {
  Tensor result1 = at::empty({0}, self.options());
  Tensor result2 = at::empty({0}, self.options());
  return at::native::std_mean_out(result1, result2, self, unbiased);
}

std::tuple<Tensor,Tensor> var_mean(const Tensor& self, bool unbiased) {
  Tensor result1 = at::empty({0}, self.options());
  Tensor result2 = at::empty({0}, self.options());
  return at::native::var_mean_out(result1, result2, self, unbiased);
}

Tensor var(const Tensor& self, bool unbiased) {
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "var only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "var only supports strided layout, got: ", self.layout());
  TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
              "var only supports floating-point dtypes");
  auto trivial_return = _allreduce_return_trivial(self, std::numeric_limits<double>::quiet_NaN());
  return trivial_return.has_value() ? trivial_return.value() : at::_var(self, unbiased);
}

Tensor var(const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::var_out(result, self, dim, unbiased, keepdim);
}

Tensor &var_out(Tensor &result, const Tensor &self, IntArrayRef dim, bool unbiased, bool keepdim) {
  return std_var_out(result, self, dim, unbiased, keepdim, false);
}

Tensor std(const Tensor& self, bool unbiased) {
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "std only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "std only supports strided layout, got: ", self.layout());
  TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
              "std only supports floating-point dtypes");
  auto trivial_return = _allreduce_return_trivial(self, std::numeric_limits<double>::quiet_NaN());
  return trivial_return.has_value() ? trivial_return.value() : at::_std(self, unbiased);
}

Tensor std(const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::std_out(result, self, dim, unbiased, keepdim);
}

Tensor &std_out(Tensor &result, const Tensor &self, IntArrayRef dim, bool unbiased, bool keepdim) {
  return std_var_out(result, self, dim, unbiased, keepdim, true);
}

Tensor std(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
  return  at::std(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor& std_out(Tensor& result, const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
  return at::std_out(result, self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor var(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
  return  at::var(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor& var_out(Tensor& result, const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
  return at::std_out(result, self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

std::tuple<Tensor,Tensor> var_mean(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
  return at::var_mean(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

std::tuple<Tensor,Tensor> std_mean(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
  return at::std_mean(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor& norm_out(Tensor& result, const Tensor& self, optional<Scalar> p, DimnameList dim, bool keepdim, ScalarType dtype) {
  return at::norm_out(result, self, p, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor& norm_out(Tensor& result, const Tensor& self, optional<Scalar> p, DimnameList dim, bool keepdim) {
  return at::norm_out(result, self, p, dimnames_to_positions(self, dim), keepdim);
}

Tensor norm(const Tensor& self, optional<Scalar> p, DimnameList dim, bool keepdim, ScalarType dtype) {
  return at::norm(self, p, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor norm(const Tensor& self, optional<Scalar> p, DimnameList dim, bool keepdim) {
  return at::norm(self, p, dimnames_to_positions(self, dim), keepdim);
}

Tensor any(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("any");
}
Tensor& any_out(Tensor& result, const Tensor &self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("any");
}
Tensor all(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("all");
}
Tensor& all_out(Tensor& result, const Tensor &self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("all");
}
Tensor cumsum(const Tensor& self, Dimname dim, c10::optional<ScalarType> dtype) {
  return at::cumsum(self, dimname_to_position(self, dim), dtype);
}
Tensor& cumsum_out(Tensor& result, const Tensor& self, Dimname dim, c10::optional<ScalarType> dtype) {
  return at::cumsum_out(result, self, dimname_to_position(self, dim), dtype);
}
Tensor cumprod(const Tensor& self, Dimname dim, c10::optional<ScalarType> dtype) {
  return at::cumprod(self, dimname_to_position(self, dim), dtype);
}
Tensor& cumprod_out(Tensor& result, const Tensor& self, Dimname dim, c10::optional<ScalarType> dtype) {
  return at::cumprod_out(result, self, dimname_to_position(self, dim), dtype);
}
std::tuple<Tensor, Tensor> cummax(const Tensor& self, Dimname dim) {
  return at::cummax(self, dimname_to_position(self, dim));
}
std::tuple<Tensor&, Tensor&> cummax_out(Tensor& values, Tensor& indices, const Tensor& self, Dimname dim) {
  return at::cummax_out(values, indices, self, dimname_to_position(self, dim));
}
std::tuple<Tensor, Tensor> cummin(const Tensor& self, Dimname dim) {
  return at::cummin(self, dimname_to_position(self, dim));
}
std::tuple<Tensor&, Tensor&> cummin_out(Tensor& values, Tensor& indices, const Tensor& self, Dimname dim) {
  return at::cummin_out(values, indices, self, dimname_to_position(self, dim));
}

Tensor dist(const Tensor &self, const Tensor& other, Scalar p){
  return at::norm(self - other, p);
}

}} // namespace at::native
