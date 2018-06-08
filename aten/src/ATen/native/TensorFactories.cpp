// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include "ATen/ATen.h"
#include "ATen/CPUGenerator.h"
#include "ATen/CheckGenerator.h"
#include "ATen/Dispatch.h"
#include "ATen/Error.h"
#include "ATen/NativeFunctions.h"
#include "ATen/ScalarType.h"
#include "TH/THRandom.h"

#include <algorithm>
#include <sstream>
#include <cmath>

namespace at {
namespace native {

Tensor arange(const Type& dtype, Scalar start, Scalar end, Scalar step) {
  return dtype._arange(start, end, step);
}

Tensor& arange_out(Tensor& result, Scalar start, Scalar end, Scalar step) {
  return at::_arange_out(result, start, end, step);
}

Tensor arange(const Type& dtype, Scalar end) {
  return dtype._arange(end);
}

Tensor& arange_out(Tensor& result, Scalar end) {
  return at::_arange_out(result, end);
}

Tensor empty(const Type& dtype, IntList size) {
  return dtype.tensor(size);
}

Tensor& empty_out(Tensor& result, IntList size) {
  if (result.is_sparse()) {
    result.sparse_raw_resize_(size, size.size(), 0);
  } else {
    result.resize_(size);
  }
  return result;
}

// Temporary type cast operators. These are needed to trace type-casts now since
// Type's are not supported in the IR. Instead, we call down to these
// specialized operators for each datatype.
// TODO: remove when we have Type support in the IR

#define DEFINE_CAST_OP(_1, n, _2)                                \
  Tensor _cast_##n(const Tensor& self, bool non_blocking) {      \
    auto& target_type = self.type().toScalarType(ScalarType::n); \
    if (self.type() == target_type)                              \
      return self;                                               \
    return target_type.copy(self, non_blocking);                 \
  }

AT_FORALL_SCALAR_TYPES(DEFINE_CAST_OP)

#undef DEFINE_CAST_OP

Tensor empty_like(const Tensor& self) {
  return at::native::empty_like(self, self.type());
}

Tensor empty_like(const Tensor& self, const Type& dtype) {
  if (dtype.is_sparse() && self.type().is_sparse()) {
    auto res = dtype.tensor();
    // resize_as_ requires the same exact type.
    res.sparse_raw_resize_(self.sizes(), self._dimI(), self._dimV());
    return res;
  }
  return at::native::empty(dtype, self.sizes());
}

Tensor eye(const Type& dtype, int64_t n, int64_t m) {
  auto result = dtype.tensor();
  return at::eye_out(result, n, m);
}

Tensor& eye_out_cpu(Tensor& result, int64_t n, int64_t m) {
  if (n <= 0) {
    std::ostringstream oss;
    oss << "n must be greater than 0, got: " << n;
    throw std::runtime_error(oss.str());
  }
  if(m <= 0) {
    m = n;
  }

  result.resize_({n, m});
  result.zero_();

  int64_t sz = std::min<int64_t>(n, m);
  AT_DISPATCH_ALL_TYPES(result.type(), "eye", [&]() -> void {
    scalar_t* result_data = result.data<scalar_t>();
    for(int64_t i = 0; i < sz; i++) {
      result_data[i*(result.strides()[0] + result.strides()[1])] = 1;
    }
  });

  return result;
}

Tensor full(const Type& dtype, IntList size, Scalar fill_value) {
  if (dtype.is_sparse()) {
    AT_ERROR("full(...) is not implemented for sparse types, got: ", dtype.toString());
  }
  auto result = dtype.tensor(size);
  return result.fill_(fill_value);
}

Tensor& full_out(Tensor& result, IntList size, Scalar fill_value) {
  if (result.is_sparse()) {
    AT_ERROR("full(...) is not implemented for sparse types, got: ", result.type().toString());
  }
  result.resize_(size);
  return result.fill_(fill_value);
}

Tensor full_like(const Tensor& self, Scalar fill_value) {
  return at::native::full_like(self, fill_value, self.type());
}

Tensor full_like(const Tensor& self, Scalar fill_value, const Type& dtype) {
  return at::native::full(dtype, self.sizes(), fill_value);
}

Tensor linspace(const Type& dtype, Scalar start, Scalar end, int64_t steps) {
  return dtype._linspace(start, end, steps);
}

Tensor& linspace_out(Tensor& result, Scalar start, Scalar end, int64_t steps) {
  return at::_linspace_out(result, start, end, steps);
}

Tensor logspace(const Type& dtype, Scalar start, Scalar end, int64_t steps) {
  return dtype._logspace(start, end, steps);
}

Tensor& logspace_out(Tensor& result, Scalar start, Scalar end, int64_t steps) {
  return at::_logspace_out(result, start, end, steps);
}

Tensor ones(const Type& dtype, IntList size) {
  auto result = dtype.tensor(size);
  return result.fill_(1);
}

Tensor& ones_out(Tensor& result, IntList size) {
  result.resize_(size);
  return result.fill_(1);
}

Tensor ones_like(const Tensor& self) {
  return at::native::ones(self.type(), self.sizes());
}

Tensor ones_like(const Tensor& self, const Type& dtype) {
  return at::native::ones(dtype, self.sizes());
}

Tensor rand(const Type& dtype, IntList size, Generator* generator) {
  Tensor result = dtype.tensor(size);
  return result.uniform_(0, 1, generator);
}

Tensor& rand_out(Tensor& result, IntList size, Generator* generator) {
  result.resize_(size);
  return result.uniform_(0, 1, generator);
}

Tensor rand_like(const Tensor& self) {
  return at::native::rand_like(self, self.type());
}

Tensor rand_like(const Tensor& self, const Type& dtype) {
  return at::native::rand(dtype, self.sizes());
}

Tensor randint(const Type& dtype, int64_t high, IntList size, Generator* generator) {
  Tensor result = dtype.tensor(size);
  return result.random_(0, high, generator);
}

Tensor randint(const Type& dtype, int64_t low, int64_t high, IntList size, Generator* generator) {
  Tensor result = dtype.tensor(size);
  return result.random_(low, high, generator);
}

Tensor& randint_out(Tensor& result, int64_t high, IntList size, Generator* generator) {
  result.resize_(size);
  return result.random_(0, high, generator);
}

Tensor& randint_out(Tensor& result, int64_t low, int64_t high, IntList size, Generator* generator) {
  result.resize_(size);
  return result.random_(low, high, generator);
}

Tensor randint_like(const Tensor& self, int64_t high) {
  return at::native::randint_like(self, high, self.type());
}

Tensor randint_like(const Tensor& self, int64_t low, int64_t high) {
  return at::native::randint_like(self, low, high, self.type());
}

Tensor randint_like(const Tensor& self, int64_t high, const Type& dtype) {
  return at::native::randint(dtype, high, self.sizes(), nullptr);
}

Tensor randint_like(const Tensor& self, int64_t low, int64_t high, const Type& dtype) {
  return at::native::randint(dtype, low, high, self.sizes(), nullptr);
}

Tensor randn(const Type& dtype, IntList size, Generator* generator) {
  Tensor result = dtype.tensor(size);
  return result.normal_(0, 1, generator);
}

Tensor& randn_out(Tensor& result, IntList size, Generator* generator) {
  result.resize_(size);
  return result.normal_(0, 1, generator);
}

Tensor randn_like(const Tensor& self) {
  return at::native::randn_like(self, self.type());
}

Tensor randn_like(const Tensor& self, const Type& dtype) {
  return at::native::randn(dtype, self.sizes(), nullptr);
}

namespace {
template <typename scalar_t>
void randperm_cpu(Tensor& result, int64_t n, THGenerator* generator) {
  scalar_t *r__data = result.data<scalar_t>();

  result.resize_({n});
  int64_t r__stride_0 = result.stride(0);

  for(int64_t i = 0; i < n; i++) {
    r__data[i*r__stride_0] = static_cast<scalar_t>(i);
  }

  for(int64_t i = 0; i < n - 1; i++)
  {
    int64_t z = THRandom_random(generator) % (n-i);
    scalar_t sav = r__data[i*r__stride_0];
    r__data[i*r__stride_0] = r__data[(z+i)*r__stride_0];
    r__data[(z+i)*r__stride_0] = sav;
  }
}
} // namespace


THGenerator* get_generator(at::Generator* gen) {
  auto default_gen = &at::globalContext().defaultGenerator(at::Backend::CPU);
  auto gen_ = at::check_generator<at::CPUGenerator>(gen, default_gen);
  return gen_->generator;
}

Tensor randperm(const Type& dtype, int64_t n, Generator* generator) {
  Tensor result = dtype.tensor(n);
  return at::randperm_out(result, n, generator);
}

Tensor& randperm_out_cpu(Tensor& result, int64_t n, Generator* generator) {
  if (n < 0) {
    std::ostringstream oss;
    oss << "n must be non-negative, got " << n;
    throw std::runtime_error(oss.str());
  }

  result.resize_({n});
  auto gen = get_generator(generator);
  AT_DISPATCH_ALL_TYPES(result.type(), "randperm", [&]() -> void {
    randperm_cpu<scalar_t>(result, n, gen);
  });

  return result;
}

Tensor range(const Type& dtype, Scalar start, Scalar end, Scalar step) {
  return dtype._range(start, end, step);
}

Tensor& range_out(Tensor& result, Scalar start, Scalar end, Scalar step) {
  return at::_range_out(result, start, end, step);
}

Tensor zeros(const Type& dtype, IntList size) {
  auto result = dtype.tensor(size);
  return at::native::zeros_out(result, size);
}

Tensor& zeros_out(Tensor& result, IntList size) {
  if (result.is_sparse()) {
    result.sparse_raw_resize_(size, size.size(), 0);
  } else {
    result.resize_(size);
  }
  return result.zero_();
}

Tensor zeros_like(const Tensor& self) {
  return at::native::zeros_like(self, self.type());
}

Tensor zeros_like(const Tensor& self, const Type& dtype) {
  if (dtype.is_sparse() && self.type().is_sparse()) {
    auto res = dtype.tensor();
    // resize_as_ requires the same exact type.
    res.sparse_raw_resize_(self.sizes(), self._dimI(), self._dimV());
    return res;
  }
  return at::native::zeros(dtype, self.sizes());
}

// Signal Processing Window Functions

Tensor bartlett_window(const Type& dtype, int64_t window_length, bool periodic) {
  if (dtype.is_sparse()) {
    AT_ERROR("bartlett_window(...) is not implemented for sparse types, got: ", dtype.toString());
  }
  if (!at::isFloatingType(dtype.scalarType())) {
    AT_ERROR("bartlett_window(...) expects floating point dtypes, got: ", dtype.toString());
  }
  if (window_length <= 0) {
    AT_ERROR("bartlett_window(...) requires positive window_length, got window_length=%lld", window_length);
  }
  if (window_length == 1) {
    return at::ones(dtype, {1});
  }
  if (periodic) {
    window_length += 1;
  }
  auto window = at::arange(dtype, window_length).mul_(2. / static_cast<double>(window_length - 1));
  int64_t first_half_size = ((window_length - 1) >> 1) + 1;
  window.narrow(0, first_half_size, window_length - first_half_size).mul_(-1).add_(2);
  return periodic ? window.narrow(0, 0, window_length - 1) : window;
}

Tensor blackman_window(const Type& dtype, int64_t window_length, bool periodic) {
  if (dtype.is_sparse()) {
    AT_ERROR("blackman_window(...) is not implemented for sparse types, got: ", dtype.toString());
  }
  if (!at::isFloatingType(dtype.scalarType())) {
    AT_ERROR("blackman_window(...) expects floating point dtypes, got: ", dtype.toString());
  }
  if (window_length <= 0) {
    AT_ERROR("blackman_window(...) requires positive window_length, got window_length=%lld", window_length);
  }
  if (window_length == 1) {
    return at::ones(dtype, {1});
  }
  if (periodic) {
    window_length += 1;
  }
  // from https://en.wikipedia.org/wiki/Window_function#Blackman_window
  auto window = at::arange(dtype, window_length).mul_(M_PI / static_cast<double>(window_length - 1));
  window = window.mul(4).cos_().mul_(0.08) - window.mul(2).cos_().mul_(0.5) + 0.42;
  return periodic ? window.narrow(0, 0, window_length - 1) : window;
}

Tensor hamming_window(const Type& dtype, int64_t window_length, bool periodic, double alpha, double beta) {
  if (dtype.is_sparse()) {
    AT_ERROR("hamming_window(...) is not implemented for sparse types, got: ", dtype.toString());
  }
  if (!at::isFloatingType(dtype.scalarType())) {
    AT_ERROR("hamming_window(...) expects floating point dtypes, got: ", dtype.toString());
  }
  if (window_length <= 0) {
    AT_ERROR("hamming_window(...) requires positive window_length, got window_length=%lld", window_length);
  }
  if (window_length == 1) {
    return at::ones(dtype, {1});
  }
  if (periodic) {
    window_length += 1;
  }
  auto window = at::arange(dtype, window_length);
  window.mul_(M_PI * 2. / static_cast<double>(window_length - 1)).cos_().mul_(-beta).add_(alpha);
  return periodic ? window.narrow(0, 0, window_length - 1) : window;
}

Tensor hann_window(const Type& dtype, int64_t window_length, bool periodic) {
  if (dtype.is_sparse()) {
    AT_ERROR("hann_window(...) is not implemented for sparse types, got: ", dtype.toString());
  }
  if (!at::isFloatingType(dtype.scalarType())) {
    AT_ERROR("hann_window(...) expects floating point dtypes, got: ", dtype.toString());
  }
  if (window_length <= 0) {
    AT_ERROR("hann_window(...) requires positive window_length, got window_length=%lld", window_length);
  }
  return at::native::hamming_window(dtype, window_length, periodic, 0.5, 0.5);
}

}
}
