#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/util/Exception.h>
#include <ATen/native/cpu/TensorCompareKernel.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/core/EnableNamedTensor.h>

namespace {
template <typename scalar_t>
void where_cpu(
    at::Tensor& ret,
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  auto iter = at::TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(ret);
  iter.add_input(condition);
  iter.add_input(self);
  iter.add_input(other);
  iter.dont_compute_common_dtype();
  iter.build();
  if (condition.scalar_type() == at::ScalarType::Byte) {
    at::native::cpu_kernel(
      iter,
      [=](uint8_t cond_val, scalar_t self_val, scalar_t other_val) -> scalar_t {
        return cond_val ? self_val : other_val;
      });
  } else {
    at::native::cpu_kernel(
      iter,
      [=](bool cond_val, scalar_t self_val, scalar_t other_val) -> scalar_t {
        return cond_val ? self_val : other_val;
      });
  }
}
} // namespace

namespace at { namespace native {

DEFINE_DISPATCH(max_kernel);
DEFINE_DISPATCH(min_kernel);

bool allclose(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
  return at::isclose(self, other, rtol, atol, equal_nan).all().item<uint8_t>();
}

Tensor isclose(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
  // TODO: use bitwise operator overloads once we add them

  TORCH_CHECK(self.scalar_type() == other.scalar_type(), self.scalar_type(), " did not match ", other.scalar_type())

  auto actual_error = (self - other).abs();
  auto max_error = atol + rtol * other.abs();

  // `max_error` could be a float or double depending on the type of the input
  // tensors.
  // Specifically, if other is an int tensor, multiplying by rtol results in
  // float tensor.
  // It is also possible for parameters to be 'wrapped_number's, in which case
  // max_error could be promoted to double when actual error is still a float.
  if (actual_error.scalar_type() != max_error.scalar_type()) {
    actual_error = actual_error.to(max_error.scalar_type());
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
  } else if (localScalar.isIntegral(false)){
    return localScalar.to<int64_t>() != 0;
  } else if (localScalar.isBoolean()) {
    return localScalar.to<bool>();
  }
  AT_ERROR("expected non-Tensor backed scalar");
}

Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other) {
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

Tensor _s_where_cpu(const Tensor& condition, const Tensor& self, const Tensor& other) {
  Tensor ret = at::empty(self.sizes(), self.options());
  AT_DISPATCH_ALL_TYPES(ret.scalar_type(), "where_cpu", [&] {
    where_cpu<scalar_t>(ret, condition, self, other);
  });
  return ret;
}

std::tuple<Tensor, Tensor> mode(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  return at::native::mode_out(values, indices, self, dim, keepdim);
}

std::tuple<Tensor &,Tensor &> mode_out(Tensor& values, Tensor& indices,
                                       const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "mode only supports CPU AND CUDA backend, got: ", toString(self.type().backend()));
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial_no_ident(values, self, dim, keepdim, "mode")) {
    AT_ASSERT(values.dim() == 0);
    indices.resize_({}).fill_(0);
    return std::forward_as_tuple(values, indices);
  } else {
    return at::_mode_out(values, indices, self, dim, keepdim);
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

std::tuple<Tensor &,Tensor &> max_out(Tensor& max, Tensor& max_indices,
                                      const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "max only supports CPU AND CUDA backend, got: ", toString(self.type().backend()));
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

std::tuple<Tensor &,Tensor &> min_out(Tensor& min, Tensor& min_indices,
                                      const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "min only supports CPU AND CUDA backend, got: ", toString(self.type().backend()));
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

// argmax and argmin

Tensor argmax(const Tensor& self, c10::optional<int64_t> dim, bool keepdim) {
  if (dim)
    return std::get<1>(self.max(dim.value(), keepdim));
  return std::get<1>(self.reshape({-1}).max(/*dim=*/0));
}

Tensor argmin(const Tensor& self, c10::optional<int64_t> dim, bool keepdim) {
  if (dim)
    return std::get<1>(self.min(dim.value(), keepdim));
  return std::get<1>(self.reshape({-1}).min(/*dim=*/0));
}

#ifdef BUILD_NAMEDTENSOR
// Named tensor overloads

std::tuple<Tensor, Tensor> min(const Tensor& self, Dimname dim, bool keepdim) {
  TORCH_CHECK(false, "NYI: min with names");
  return at::min(self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor &,Tensor &> min_out(Tensor& min, Tensor& min_indices,
                                      const Tensor& self, Dimname dim, bool keepdim) {
  TORCH_CHECK(false, "NYI: min with names");
  return at::min_out(min, min_indices, self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor, Tensor> max(const Tensor& self, Dimname dim, bool keepdim) {
  TORCH_CHECK(false, "NYI: max with names");
  return at::max(self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor &,Tensor &> max_out(Tensor& max, Tensor& max_indices,
                                      const Tensor& self, Dimname dim, bool keepdim) {
  TORCH_CHECK(false, "NYI: max with names");
  return at::max_out(max, max_indices, self, dimname_to_position(self, dim), keepdim);
}
#endif

}} // namespace at::native
