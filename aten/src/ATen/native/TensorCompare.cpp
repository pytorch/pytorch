#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/util/Exception.h>
#include <ATen/native/cpu/TensorCompareKernel.h>
#include <ATen/NamedTensorUtils.h>


namespace at { namespace native {

DEFINE_DISPATCH(where_kernel);
DEFINE_DISPATCH(max_kernel);
DEFINE_DISPATCH(min_kernel);

bool allclose(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
  return at::isclose(self, other, rtol, atol, equal_nan).all().item<uint8_t>();
}

Tensor isclose(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
  // TODO: use bitwise operator overloads once we add them

  TORCH_CHECK(self.scalar_type() == other.scalar_type(), self.scalar_type(), " did not match ", other.scalar_type())

  // The original formula `atol + rtol * other.abs()` works incorrectly when
  // `other` has integral dtype and `other == min_value` and `abs(min_value)` is negative:
  // std::abs(std::numeric_limits<int64_t>::lowest()) == std::numeric_limits<int64_t>::lowest() < 0
  auto max_error = atol + (rtol * other).abs();

  // `max_error` could be a float or double depending on the type of the input
  // tensors.
  // Specifically, if other is an int tensor, multiplying by rtol results in
  // float tensor.
  // It is also possible for parameters to be 'wrapped_number's, in which case
  // max_error could be promoted to double when actual error is still a float.
  Tensor actual_error;
  if (actual_error.scalar_type() != max_error.scalar_type()) {
    // To silence ASAN that does not like (x - std::numeric_limits<int64_t>::lowest())
    actual_error = (self - other.to(max_error.scalar_type())).abs();
  } else {
    actual_error = (self - other).abs();
  }

  auto close = actual_error <= max_error;

  if (isFloatingType(self.scalar_type()) && isFloatingType(other.scalar_type())) {
    // Handle +/-inf
    close.__ior__(self == other);
    close.__iand__((self == INFINITY) == (other == INFINITY));
    close.__iand__((self == -INFINITY) == (other == -INFINITY));

    if (equal_nan) {
      close.__ior__((self != self).__and__((other != other)));
    }
  }
  return close;
}

Tensor isnan(const Tensor& self) {
  return self != self;
}

Tensor isinf(const Tensor &self) {
  // Integral tensor types are always not inf
  if (isIntegralType(self.scalar_type())) {
    return at::zeros_like(self, at::kBool, at::MemoryFormat::Preserve);
  }
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "isinf", [&]() {
    return self.abs() == std::numeric_limits<scalar_t>::infinity();
  });
}

Tensor isfinite(const Tensor& self) {
  // Integral tensor types are finite
  if (!self.is_floating_point()) {
    return at::ones_like(self, at::kBool, at::MemoryFormat::Preserve);
  }
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "isfinite", [&]() {
    return (self == self) * (self.abs() != std::numeric_limits<scalar_t>::infinity());
  });
}

bool is_nonzero(const Tensor& self) {
  auto n = self.numel();
  AT_ASSERT(n >= 0);
  if (n == 0) {
    AT_ERROR("bool value of Tensor with no values is ambiguous");
  }
  if (n > 1) {
    AT_ERROR("bool value of Tensor with more than one value is ambiguous");
  }
  Scalar localScalar = self.item();
  if (localScalar.isFloatingPoint()) {
    return localScalar.to<double>() != 0;
  } else if (localScalar.isComplex()) {
     return localScalar.to<std::complex<double>>() != std::complex<double>(0.0, 0.0);
  } else if (localScalar.isIntegral(false)){
    return localScalar.to<int64_t>() != 0;
  } else if (localScalar.isBoolean()) {
    return localScalar.to<bool>();
  }
  AT_ERROR("expected non-Tensor backend scalar");
}

Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(condition.device() == self.device() && self.device() == other.device(),
              "expected condition, x and y to be on the same device, but condition is on ",
              condition.device(), " and x and y are on ", self.device(), " and ", other.device(),
              " respectively");
  if (condition.scalar_type() != ScalarType::Byte && condition.scalar_type() != ScalarType::Bool) {
    AT_ERROR("Expected condition to have ScalarType Byte, but got ScalarType ",
                  toString(condition.scalar_type()));
  }
  Tensor b_condition, b_self, b_other;
  std::tie(b_condition, b_self, b_other) = expand_outplace(condition, self, other, "where");
  return at::_s_where(b_condition, b_self, b_other);
}

std::vector<Tensor> where(const Tensor& condition) {
  return condition.nonzero_numpy();
}

Tensor _s_where(const Tensor& condition, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(self.dtype() == other.dtype(), "expected scalar type ", self.dtype(), " but found ", other.dtype());
  Tensor ret = at::empty(self.sizes(), self.options());
  auto iter = at::TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(ret);
  iter.add_input(condition);
  iter.add_input(self);
  iter.add_input(other);
  iter.dont_compute_common_dtype();
  iter.build();
  where_kernel(iter.device_type(), iter, condition.scalar_type());
  return ret;
}

std::tuple<Tensor, Tensor> mode(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  return at::native::mode_out(values, indices, self, dim, keepdim);
}

std::tuple<Tensor &,Tensor &> mode_out(Tensor& values, Tensor& indices,
                                       const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "mode only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "mode only supports strided layout, got: ", self.layout());
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial_no_ident(values, self, dim, keepdim, "mode")) {
    AT_ASSERT(values.dim() == 0);
    indices.resize_({}).fill_(0);
    return std::forward_as_tuple(values, indices);
  } else {
    auto result = [&]() {
      NoNamesGuard guard;
      return at::_mode_out(values, indices, self, dim, keepdim);
    }();
    namedinference::propagate_names_for_reduction(std::get<0>(result), self, dim, keepdim);
    namedinference::propagate_names_for_reduction(std::get<1>(result), self, dim, keepdim);
    return result;
  }
}

std::tuple<Tensor &,Tensor &> _max_out_cpu(Tensor& max, Tensor& max_indices,
                                        const Tensor& self, int64_t dim, bool keepdim) {
  if (self.is_contiguous() && max.is_contiguous() && max_indices.is_contiguous()) {
    _dimreduce_setup(max, self, dim);
    _dimreduce_setup(max_indices, self, dim);
    max_kernel(kCPU, max, max_indices, self, dim);
    if (!keepdim) {
      max.squeeze_(dim);
      max_indices.squeeze_(dim);
    }
    return std::tuple<Tensor &,Tensor &>{max, max_indices};
  }
  return at::_max_out(max, max_indices, self, dim, keepdim);
}

std::tuple<Tensor, Tensor> max(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor max_indices = at::empty({0}, self.options().dtype(kLong));
  if (self.is_quantized()) {
    Tensor max = at::empty({0}, self.options().dtype(toUnderlying(self.scalar_type())));
    at::native::max_out(max, max_indices, self.int_repr(), dim, keepdim);
    // TODO: qscheme
    return std::tuple<Tensor, Tensor>(at::_make_per_tensor_quantized_tensor(max, self.q_scale(), self.q_zero_point()), max_indices);
  } else {
    Tensor  max = at::empty({0}, self.options());
    return at::native::max_out(max, max_indices, self, dim, keepdim);
  }
}

static std::tuple<Tensor &,Tensor &> max_out_impl(Tensor& max, Tensor& max_indices,
                                                  const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "max only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "max only supports strided layout, got: ", self.layout());
  TORCH_CHECK(self.device() == max.device(),
              "expected device ", self.device(), " but got ",
              max.device(), " for max values output");
  TORCH_CHECK(self.device() == max_indices.device(),
              "expected device ", self.device(), " but got ",
              max_indices.device(), " for indices output");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial_no_ident(max, self, dim, keepdim, "max")) {
    AT_ASSERT(max.dim() == 0);
    max_indices.resize_({}).fill_(0);
    return std::forward_as_tuple(max, max_indices);
  } else {
    if (self.is_cuda()) {
      return at::_max_out(max, max_indices, self, dim, keepdim);
    } else {
      return _max_out_cpu(max, max_indices, self, dim, keepdim);
    }
  }
}

std::tuple<Tensor&,Tensor&> max_out(Tensor& max, Tensor& max_indices,
                                      const Tensor& self, int64_t dim, bool keepdim) {
  auto result = [&]() {
    NoNamesGuard guard;
    return max_out_impl(max, max_indices, self, dim, keepdim);
  }();
  namedinference::propagate_names_for_reduction(max, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(max_indices, self, dim, keepdim);
  return result;
}

std::tuple<Tensor &,Tensor &> _min_out_cpu(Tensor& min, Tensor& min_indices,
                                        const Tensor& self, int64_t dim, bool keepdim) {
  if (self.is_contiguous() && min.is_contiguous() && min_indices.is_contiguous()) {
    _dimreduce_setup(min, self, dim);
    _dimreduce_setup(min_indices, self, dim);
    min_kernel(kCPU, min, min_indices, self, dim);
    if (!keepdim) {
      min.squeeze_(dim);
      min_indices.squeeze_(dim);
    }
    return std::tuple<Tensor &,Tensor &>{min, min_indices};
  }
  return at::_min_out(min, min_indices, self, dim, keepdim);
}

std::tuple<Tensor, Tensor> min(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor min_indices = at::empty({0}, self.options().dtype(kLong));
  if (self.is_quantized()) {
    Tensor min = at::empty({0}, self.options().dtype(toUnderlying(self.scalar_type())));
    at::native::min_out(min, min_indices, self.int_repr(), dim, keepdim);
    return std::tuple<Tensor, Tensor>(at::_make_per_tensor_quantized_tensor(min, self.q_scale(), self.q_zero_point()), min_indices);
  } else {
    Tensor min = at::empty({0}, self.options());
    return at::native::min_out(min, min_indices, self, dim, keepdim);
  }
}

static std::tuple<Tensor &,Tensor &> min_out_impl(Tensor& min, Tensor& min_indices,
                                                  const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "min only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "min only supports strided layout, got: ", self.layout());
  TORCH_CHECK(self.device() == min.device(),
              "expected device ", self.device(), " but got ",
              min.device(), " for min values output");
  TORCH_CHECK(self.device() == min_indices.device(),
              "expected device ", self.device(), " but got ",
              min_indices.device(), " for indices output");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial_no_ident(min, self, dim, keepdim, "min")) {
    AT_ASSERT(min.dim() == 0);
    min_indices.resize_({}).fill_(0);
    return std::forward_as_tuple(min, min_indices);
  } else {
    if (self.is_cuda()) {
      return at::_min_out(min, min_indices, self, dim, keepdim);
    } else {
      return _min_out_cpu(min, min_indices, self, dim, keepdim);
    }
  }
}

std::tuple<Tensor&,Tensor&> min_out(Tensor& min, Tensor& min_indices,
                                    const Tensor& self, int64_t dim, bool keepdim) {
  auto result = [&]() {
    NoNamesGuard guard;
    return min_out_impl(min, min_indices, self, dim, keepdim);
  }();
  namedinference::propagate_names_for_reduction(min, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(min_indices, self, dim, keepdim);
  return result;
}


// Named tensor overloads

std::tuple<Tensor, Tensor> min(const Tensor& self, Dimname dim, bool keepdim) {
  return at::min(self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor &,Tensor &> min_out(Tensor& min, Tensor& min_indices,
                                      const Tensor& self, Dimname dim, bool keepdim) {
  return at::min_out(min, min_indices, self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor, Tensor> max(const Tensor& self, Dimname dim, bool keepdim) {
  return at::max(self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor &,Tensor &> max_out(Tensor& max, Tensor& max_indices,
                                      const Tensor& self, Dimname dim, bool keepdim) {
  return at::max_out(max, max_indices, self, dimname_to_position(self, dim), keepdim);
}
Tensor argmax(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("argmax");
}
Tensor argmin(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("argmin");
}
Tensor argsort(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("argsort");
}
std::tuple<Tensor, Tensor> mode(const Tensor& self, Dimname dim, bool keepdim) {
  return at::mode(self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor &,Tensor &> mode_out(Tensor& values, Tensor& indices,
                                       const Tensor& self, Dimname dim, bool keepdim) {
  return at::mode_out(values, indices, self, dimname_to_position(self, dim), keepdim);
}

}} // namespace at::native
