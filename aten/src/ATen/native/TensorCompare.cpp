#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctions.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/util/Exception.h>
#include <ATen/native/cpu/TensorCompareKernel.h>

namespace {
template <typename scalar_t>
void where_cpu(
    at::Tensor& ret,
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  at::CPU_tensor_apply4<scalar_t, uint8_t, scalar_t, scalar_t>(
      ret,
      condition,
      self,
      other,
      [](scalar_t& ret_val,
         const uint8_t& cond_val,
         const scalar_t& self_val,
         const scalar_t& other_val) {
        ret_val = cond_val ? self_val : other_val;
      });
}
} // namespace

namespace at { namespace native {

DEFINE_DISPATCH(max_kernel);
DEFINE_DISPATCH(min_kernel);

Tensor index_select_backward(const Tensor& grad, int64_t dim, const Tensor& indices, IntArrayRef sizes, bool keepdim) {
  Tensor res = at::zeros(sizes, grad.options());
  if (!keepdim && sizes.size() > 0) {
    return res.scatter_(dim, indices.unsqueeze(dim), grad.unsqueeze(dim));
  }
  return res.scatter_(dim, indices, grad);
}

bool allclose(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
  return at::isclose(self, other, rtol, atol, equal_nan).all().item<uint8_t>();
}

Tensor isclose(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
  // TODO: use bitwise operator overloads once we add them
  auto actual_error = (self - other).abs();
  auto max_error = atol + rtol * other.abs();
  auto close = actual_error <= max_error;

  if (isFloatingType(self.type().scalarType()) && isFloatingType(other.type().scalarType())) {
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
  } else if (localScalar.isIntegral()){
    return localScalar.to<int64_t>() != 0;
  }
  AT_ERROR("expected non-Tensor backed scalar");
}

Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other) {
  if (condition.type().scalarType() != ScalarType::Byte) {
    AT_ERROR("Expected condition to have ScalarType Byte, but got ScalarType ",
                  toString(condition.type().scalarType()));
  }
  Tensor b_condition, b_self, b_other;
  std::tie(b_condition, b_self, b_other) = expand_outplace(condition, self, other, "where");
  return at::_s_where(b_condition, b_self, b_other);
}

Tensor _s_where_cpu(const Tensor& condition, const Tensor& self, const Tensor& other) {
  Tensor ret = at::empty(self.sizes(), self.options());
  AT_DISPATCH_ALL_TYPES(ret.type(), "where", [&] {
    where_cpu<scalar_t>(ret, condition, self, other);
  });
  return ret;
}

std::tuple<Tensor, Tensor> kthvalue(const Tensor& self, int64_t k, int64_t dim, bool keepdim) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  return at::native::kthvalue_out(values, indices, self, k, dim, keepdim);
}

std::tuple<Tensor &,Tensor &> kthvalue_out(Tensor& values, Tensor& indices,
                                           const Tensor& self, int64_t k, int64_t dim, bool keepdim) {
  AT_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "kthvalue only supports CPU AND CUDA backend, got: ", toString(self.type().backend()));
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial_no_ident(values, self, dim, keepdim, "kthvalue")) {
    AT_ASSERT(values.dim() == 0);
    indices.resize_({}).fill_(0);
    return std::forward_as_tuple(values, indices);
  } else {
    return at::legacy::th::_th_kthvalue_out(values, indices, self, k, dim, keepdim);
  }
}

std::tuple<Tensor, Tensor> median(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  return at::native::median_out(values, indices, self, dim, keepdim);
}

std::tuple<Tensor &,Tensor &> median_out(Tensor& values, Tensor& indices,
                                         const Tensor& self, int64_t dim, bool keepdim) {
  AT_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "median only supports CPU AND CUDA backend, got: ", toString(self.type().backend()));
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial_no_ident(values, self, dim, keepdim, "median")) {
    AT_ASSERT(values.dim() == 0);
    indices.resize_({}).fill_(0);
    return std::forward_as_tuple(values, indices);
  } else {
    return at::legacy::th::_th_median_out(values, indices, self, dim, keepdim);
  }
}

std::tuple<Tensor, Tensor> mode(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  return at::native::mode_out(values, indices, self, dim, keepdim);
}

std::tuple<Tensor &,Tensor &> mode_out(Tensor& values, Tensor& indices,
                                       const Tensor& self, int64_t dim, bool keepdim) {
  AT_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "mode only supports CPU AND CUDA backend, got: ", toString(self.type().backend()));
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial_no_ident(values, self, dim, keepdim, "mode")) {
    AT_ASSERT(values.dim() == 0);
    indices.resize_({}).fill_(0);
    return std::forward_as_tuple(values, indices);
  } else {
    return at::legacy::th::_th_mode_out(values, indices, self, dim, keepdim);
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
  return at::legacy::th::_th_max_out(max, max_indices, self, dim, keepdim);
}

std::tuple<Tensor, Tensor> max(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor max = at::empty({0}, self.options());
  Tensor max_indices = at::empty({0}, self.options().dtype(kLong));
  return at::native::max_out(max, max_indices, self, dim, keepdim);
}

std::tuple<Tensor &,Tensor &> max_out(Tensor& max, Tensor& max_indices,
                                      const Tensor& self, int64_t dim, bool keepdim) {
  AT_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "max only supports CPU AND CUDA backend, got: ", toString(self.type().backend()));
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial_no_ident(max, self, dim, keepdim, "max")) {
    AT_ASSERT(max.dim() == 0);
    max_indices.resize_({}).fill_(0);
    return std::forward_as_tuple(max, max_indices);
  } else {
    if (self.is_cuda()) {
      return at::legacy::th::_th_max_out(max, max_indices, self, dim, keepdim);
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
  return at::legacy::th::_th_min_out(min, min_indices, self, dim, keepdim);
}

std::tuple<Tensor, Tensor> min(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor min = at::empty({0}, self.options());
  Tensor min_indices = at::empty({0}, self.options().dtype(kLong));
  return at::native::min_out(min, min_indices, self, dim, keepdim);
}

std::tuple<Tensor &,Tensor &> min_out(Tensor& min, Tensor& min_indices,
                                      const Tensor& self, int64_t dim, bool keepdim) {
  AT_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "min only supports CPU AND CUDA backend, got: ", toString(self.type().backend()));
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial_no_ident(min, self, dim, keepdim, "min")) {
    AT_ASSERT(min.dim() == 0);
    min_indices.resize_({}).fill_(0);
    return std::forward_as_tuple(min, min_indices);
  } else {
    if (self.is_cuda()) {
      return at::legacy::th::_th_min_out(min, min_indices, self, dim, keepdim);
    } else {
      return _min_out_cpu(min, min_indices, self, dim, keepdim);
    }
  }
}

// argmax and argmin

Tensor argmax(const Tensor& self, int64_t dim, bool keepdim) {
  return std::get<1>(self.max(dim, keepdim));
}

Tensor argmax(const Tensor& self) {
  return std::get<1>(self.reshape({-1}).max(/*dim=*/0));
}

Tensor argmin(const Tensor& self, int64_t dim, bool keepdim) {
  return std::get<1>(self.min(dim, keepdim));
}

Tensor argmin(const Tensor& self) {
  return std::get<1>(self.reshape({-1}).min(/*dim=*/0));
}

// `argmin` and `argmax` are exposed in C++ but not in Python, where we only
// expose `_argmin` and `_argmax` (which call the first versions). In Python,
// we then define our own `argmax` and `argmin` that handle passing `dim=None`,
// which gets the argmax/argmin of the flattened array.

Tensor _argmax(const Tensor& self, int64_t dim, bool keepdim) {
  return at::argmax(self, dim, keepdim);
}

Tensor _argmin(const Tensor& self, int64_t dim, bool keepdim) {
  return at::argmin(self, dim, keepdim);
}
}} // namespace at::native
